import glob
import copy
import logging
import numpy as np
from numba import njit
import torch
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from data.data_utils import connect_adl
from main.main_utils import scatter_tensor_from_one, all_gather


logger = logging.getLogger(__name__)


def input_ids_to_dict_feat(feat, max_special_tokenid, mlm_prob, mask_tokenid, vocab_size):
    labels = feat.clone()

    def get_mask(feat, prob):
        return torch.bernoulli(torch.full(feat.shape, prob, device=feat.device)).bool()

    # special tokens cannot be masked and cannot be considered as labels as well
    switch_tokens = (feat > max_special_tokenid) & get_mask(feat, mlm_prob)
    labels[~switch_tokens] = -100

    replace_with_mask = switch_tokens & get_mask(feat, 0.8)
    feat[replace_with_mask] = mask_tokenid

    replace_with_noise = switch_tokens & ~replace_with_mask & get_mask(feat, 0.5)
    noise_tokens = torch.randint(vocab_size, size=feat.shape, dtype=torch.int32, device=feat.device)
    feat[replace_with_noise] = noise_tokens[replace_with_noise]

    return {
        'input_ids': feat.long(),
        'labels': labels.long(),
        'attention_mask': torch.ones_like(feat).long()
    }


class BinLMLMLoader():
    '''
    Because of efficiency, loading will happen in only rank=0 worker, others will just collect data
    before starting
    '''

    prepare_fn = lambda *args: None

    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.local_world_size = kwargs.get('local_world_size', self.world_size)
        self.part_of_multitask = kwargs.get('part_of_multitask', False)

        try:
            self.num_steps = int(config['num_steps'])
        except:
            raise Exception(f'bin lines datareader needs integer steps {config["num_steps"]}')
        self.num_batches = 0
        self.batches_prepared = 0

        data_config = config['data']
        self.maxlen = data_config.get('maxlen', 32)
        self.mask_tokenid = data_config.get('mask_tokenid', 103)
        self.max_special_tokenid = data_config.get('max_special_tokenid', 998)
        self.vocab_size = data_config.get('vocab_size', 30522)
        self.mlm_prob = data_config.get('mlm_prob', 0.3)

        self.target = ['text', 'loss']  # text->loss would give final loss
        self.reader_queue_len = data_config.get('reader_queue_len', 2)
        self.num_reader_threads = data_config.get('num_reader_threads', 1)

        dl_config = config['dl']
        self.batch_size = dl_config['batch_size']
        assert self.batch_size % world_size == 0, "batch_size needs to be a multiple of world_size"
        self.batches_queue_len = dl_config.get('batches_queue_len', 100)

        if self.rank == 0:
            self.populate_all_files(data_config['file_pattern'])
            # used when joining, helps with emptying the queues and cleaning up
            self.stopped = False
            self.start_async()

    def start_async(self):
        # ~store files in memory in queue, async loader will read files when we need them
        self._file_contents_queue = Queue(self.reader_queue_len)
        self._file_reader_pool = ThreadPoolExecutor(self.num_reader_threads)
        self._reader_submitter_background = ThreadPoolExecutor(1)
        self._reader_submitter_background.submit(self._submit_file_reads_async)

        self._train_batches_queue = Queue(self.batches_queue_len)
        # used to store data which was left after everything else was batched in a file
        # but drop this data if previous file was the same
        self._data_from_prev_file, self._prev_file = None, None
        self._batcher_pool = ThreadPoolExecutor(1)
        self._batcher_pool.submit(self._create_batches_async)

    def populate_all_files(self, file_pattern):
        self.all_files = []
        for pattern in file_pattern.split('|'):
            self.all_files.extend(glob.glob(pattern))

    def _read_file(self, fpath):
        return np.fromfile(fpath, dtype=np.uint16)

    def _populate_file_contents_queue(self, file_path):
        # check whether pool has been stopped before starting file read
        if not self.stopped:
            feat = self._read_file(file_path)
            assert len(feat) % (1+self.maxlen) == 0
            feat = feat.reshape(((-1, 1+self.maxlen)))
            assert np.all(feat[:, 0] == self.maxlen)
            feat = torch.from_numpy(feat[:, 1:].astype(np.int32))
        # check whether pool has been stopped before putting data into queue
        if not self.stopped:
            # put read data in queue, this waits when queue is full until someone reads from it
            self._file_contents_queue.put((file_path, feat))

    def _submit_file_reads_async(self):
        while not self.stopped:
            np.random.shuffle(self.all_files)
            _res = self._file_reader_pool.map(self._populate_file_contents_queue, self.all_files)
            assert all(map(lambda x: x is None, _res))

    def _create_batches_async(self):
        while not self.stopped:
            file_path, feats = self._file_contents_queue.get(timeout=600)
            logger.info(f'creating batches from {file_path} already prepared batches {self.batches_prepared}')
            self._file_contents_queue.task_done()
            if self._data_from_prev_file is not None:
                if self._prev_file != file_path:
                    feats = torch.cat((feats, self._data_from_prev_file), 0)
                self._data_from_prev_file = None
                self._prev_file = None
            shuffle_order = torch.randperm(feats.shape[0])
            batches_idxs = torch.split(shuffle_order, self.batch_size, dim=0)
            if len(batches_idxs[-1]) != self.batch_size:
                self._data_from_prev_file = feats[batches_idxs[-1]]
                self._prev_file = file_path
                batches_idxs = batches_idxs[:-1]
            for batch_idxs in batches_idxs:
                if self.stopped:
                    break
                self._train_batches_queue.put(feats[batch_idxs])
                self.batches_prepared += 1
                if self.batches_prepared >= self.num_steps:
                    logger.info('stopping data sampler')
                    self.stopped = True

    def join(self):
        if self.rank != 0:
            return
        self.stopped = True

        self._reader_submitter_background.shutdown(wait=True)
        logger.info("reader submitter pool shutdown")
        while self._file_contents_queue.qsize() > 0:
            self._file_contents_queue.get(timeout=1)
            self._file_contents_queue.task_done()
        self._file_contents_queue.join()
        self._file_reader_pool.shutdown(wait=True)
        logger.info("reader pool shutdown")

        while self._train_batches_queue.qsize() > 0:
            self._train_batches_queue.get(timeout=1)
            self._train_batches_queue.task_done()
        self._train_batches_queue.join()
        self._batcher_pool.shutdown(wait=True)
        logger.info("batcher pool shutdown")

    def __iter__(self):
        while self.num_batches < self.num_steps:
            self.num_batches += 1
            feats = []
            if self.rank == 0:
                feats = self._train_batches_queue.get(timeout=600)
                self._train_batches_queue.task_done()

                # for multitask, atomic dataloaders are crated only in rank 0
                # and return the raw features
                if self.part_of_multitask:
                    yield feats
                    continue

                feats = list(feats.to(device='cuda').tensor_split(self.world_size))

            feats_dest = torch.empty((self.batch_size//self.world_size, self.maxlen), device='cuda',
                                     dtype=torch.int32)
            # feats are now loaded and can be transferred to other ranks
            if self.world_size > 1:
                feats, handler = scatter_tensor_from_one(feats_dest, feats)
                handler.wait()
            else:
                feats = feats[0]

            batch_data = {'feats': {'text': input_ids_to_dict_feat(feats,
                                                                   self.max_special_tokenid,
                                                                   self.mlm_prob,
                                                                   self.mask_tokenid,
                                                                   self.vocab_size)}}
            batch_data['target'] = self.target
            batch_data['batch_size'] = self.batch_size//self.world_size
            yield batch_data

    def __len__(self):
        return self.num_steps

    def parse_steps(self, steps):
        assert isinstance(steps, int), "bin lines datareader needs integer steps"
        return steps

    def state_dict(self):
        return {'num_batches': self.num_batches}

    def load_state_dict(self, sd):
        self.num_batches = sd['num_batches']


class BinLMLMAdlDataLoader(BinLMLMLoader):
    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.adl = connect_adl()
        super(BinLMLMAdlDataLoader, self).__init__(config, rank, world_size, device, *args, **kwargs)

    def populate_all_files(self, file_pattern):
        self.all_files = []
        for pattern in file_pattern.split('|'):
            self.all_files.extend(self.adl.glob(pattern))

    def _read_file(self, fpath):
        with self.adl.open(fpath, 'rb') as f:
            bytesdata = f.read()
        return np.frombuffer(bytesdata, dtype=np.uint16)


class BinLMLMAdlDataLoaderMultitask():
    prepare_fn = lambda *args: None
    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.config = config
        self.batch_size = self.config['dl']['batch_size']
        self.local_batch_size = self.batch_size // self.world_size

        data_config = config['data']
        self.maxlen = data_config.get('maxlen', 32)
        self.mask_tokenid = data_config.get('mask_tokenid', 103)
        self.max_special_tokenid = data_config.get('max_special_tokenid', 998)
        self.vocab_size = data_config.get('vocab_size', 30522)
        self.mlm_prob = data_config.get('mlm_prob', 0.3)
        self.target = ['text', 'loss']  # text->loss would give final loss

        try:
            self.num_steps, self.num_batches = int(config['num_steps']), 0
        except:
            raise Exception(f'bin lines datareader needs integer steps {config["num_steps"]}')

        # atomic dataloaders are only created in rank0
        if self.rank != 0:
            return

        data_config = config['data']
        self.datasets = list(data_config['files'].keys())
        self.num_datasets = len(self.datasets)

        # sample number of points at each step from each dataset
        # irritating point is this sampling depends on world_size as we sample and sync for local_batch_size
        # across processes, ideally we sample and then divide across processes
        sample_probs = np.array([data_config['files'][d]['rate'] for d in self.datasets], dtype=np.float32)
        sample_probs /= sample_probs.sum()
        logger.info(f'datasets: {self.datasets}')
        logger.info(f'datasets: {sample_probs}')
        self.order = np.random.multinomial(self.batch_size, sample_probs, size=self.num_steps)
        
        self.dls = {}
        for d in self.datasets:
            config_copy = copy.deepcopy(config)
            del config_copy['data']['files']
            config_copy['data']['reader_queue_len'] = data_config['files'][d].get('reader_queue_len', 2)
            config_copy['data']['num_reader_threads'] = data_config['files'][d].get('num_reader_threads', 1)
            config_copy['data']['maxlen'] = data_config['files'][d]['maxlen']
            config_copy['data']['file_pattern'] = data_config['files'][d]['file_pattern']
            self.dls[d] = BinLMLMAdlDataLoader(config_copy, rank, world_size, device, *args,
                                               part_of_multitask=True, **kwargs)

    def setup_atomic_dls(self):
        self.dl_iters = [iter(self.dls[x]) for x in self.datasets]
        # sample a batch from each source
        self.samples_stock = [next(it) for it in self.dl_iters]
        self.samples_stock_count = [self.batch_size for _ in range(self.num_datasets)]
        logger.info('sampled one batch from each source, starting overall sampling')

    def get_samples_from_stock(self, dataset_id, num_samples):
        stock_count = self.samples_stock_count[dataset_id]
        num_samples_taken = min(stock_count, num_samples)
        start_idx = self.batch_size-stock_count
        samples = self.samples_stock[dataset_id][start_idx:start_idx+num_samples_taken]
        self.samples_stock_count[dataset_id] -= num_samples_taken
        # if stock empty, replenish
        if self.samples_stock_count[dataset_id] == 0:
            self.samples_stock[dataset_id] = next(self.dl_iters[dataset_id])
            self.samples_stock_count[dataset_id] = self.batch_size
        if num_samples_taken < num_samples:
            remaining_samples_count = num_samples-num_samples_taken
            samples = torch.vstack([samples, self.samples_stock[dataset_id][:remaining_samples_count]])
            self.samples_stock_count[dataset_id] -= remaining_samples_count
        return samples

    def next_batch(self):
        # scatter the batch list and yield
        batch_parts = []
        for dataset_id in range(self.num_datasets):
            num_samples_needed = self.order[self.num_batches][dataset_id]
            if num_samples_needed == 0:
                continue
            batch_parts.append(self.get_samples_from_stock(dataset_id, num_samples_needed))
        return torch.vstack(batch_parts)

    def __iter__(self):
        if self.rank == 0:
            self.setup_atomic_dls()
        while self.num_batches < self.num_steps:
            feats = None
            if self.rank == 0:
                feats = self.next_batch()
                feats = list(feats.to(device='cuda').tensor_split(self.world_size))

            feats_dest = torch.empty((self.local_batch_size, self.maxlen), device='cuda',
                                    dtype=torch.int32)
            # feats are now loaded and can be transferred to other ranks
            if self.world_size > 1:
                feats, handler = scatter_tensor_from_one(feats_dest, feats)
                handler.wait()
            else:
                feats = feats[0]
            batch_data = {'feats': {'text': input_ids_to_dict_feat(feats,
                                                                   self.max_special_tokenid,
                                                                   self.mlm_prob,
                                                                   self.mask_tokenid,
                                                                   self.vocab_size)}}
            batch_data['target'] = self.target
            batch_data['batch_size'] = self.batch_size//self.world_size
            self.num_batches += 1
            yield batch_data

    def join(self):
        if self.rank != 0:
            return
        for dl in self.dls.values():
            dl.join()

    def __len__(self):
        return self.num_steps

    def parse_steps(self, steps):
        assert isinstance(steps, int), "bin lines datareader needs integer steps"
        return steps

    def state_dict(self):
        if self.rank != 0:
            return {}
        return {'num_batches': self.num_batches, 'dls': {k: v.state_dict() for k, v in self.dls.items()}}

    def load_state_dict(self, sd):
        if self.rank != 0:
            return
        self.num_batches = sd['num_batches']
        for k, v in sd['dls']:
            self.dls[k].load_state_dict(v)
