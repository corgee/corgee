import glob
import copy
import logging
import numpy as np
from numba import njit
import torch
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from data.data_utils import parse_target, connect_adl
from main.main_utils import scatter_tensor_from_one


logger = logging.getLogger(__name__)


@njit
def get_binlines_len(data):
    offset, rownum, numitems = 0, 0, data.shape[0]
    while offset < numitems:
        offset += data[offset] + 1
        rownum += 1
    return rownum


@njit
def fill_dense_array(data, densemat, maxlen=128):
    offset, rownum = 0, 0
    while offset < data.shape[0]:
        datal = data[offset]
        l = min(datal, maxlen)
        densemat[rownum, :l] = data[offset+1:offset+l+1]
        offset += datal + 1
        rownum += 1


@njit
def fill_dense_array_pairs(data, densemat1, densemat2):
    offset, numrows, maxlen1, maxlen2 = 0, densemat1.shape[0], densemat1.shape[1], densemat2.shape[1]
    for rownum in range(numrows):
        datal1 = data[offset]
        l1 = min(datal1, maxlen1)
        densemat1[rownum, :l1] = data[offset+1:offset+l1+1]
        offset += datal1 + 1

        datal2 = data[offset]
        l2 = min(datal2, maxlen2)
        densemat2[rownum, :l2] = data[offset+1:offset+l2+1]
        offset += datal2 + 1


def read_pairs(data, maxlen1, maxlen2, padding_idx):
    numrows = get_binlines_len(data)//2
    densemat1 = np.zeros((numrows, maxlen1), dtype=data.dtype)+padding_idx
    densemat2 = np.zeros((numrows, maxlen2), dtype=data.dtype)+padding_idx
    fill_dense_array_pairs(data, densemat1, densemat2)
    # another option is to store in int16 format and handle negative indices when vocab > 32k
    # but that would save 2x memory, but extra negative check in loading, risky
    return torch.from_numpy(densemat1.astype(np.int32)), torch.from_numpy(densemat2.astype(np.int32))


class BinLPairsDataLoader():
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

        try:
            self.num_steps = int(config['num_steps'])
        except:
            raise Exception(f'bin lines datareader needs integer steps {config["num_steps"]}')
        self.num_batches = 0
        self.batches_prepared = 0

        data_config = config['data']
        self.maxlen1 = data_config.get('maxlen1', 32)
        self.maxlen2 = data_config.get('maxlen2', 128)

        self.prefix1 = data_config.get('prefix1', None)
        if self.prefix1 is not None:
            self.prefix1 = torch.tensor(list(map(int, self.prefix1.split(','))), dtype=torch.int32)
        self.prefix2 = data_config.get('prefix2', None)
        if self.prefix2 is not None:
            self.prefix2 = torch.tensor(list(map(int, self.prefix2.split(','))), dtype=torch.int32)
        self.suffix1 = data_config.get('suffix1', None)
        if self.suffix1 is not None:
            self.suffix1 = torch.tensor(list(map(int, self.suffix1.split(','))), dtype=torch.int32)
        self.suffix2 = data_config.get('suffix2', None)
        if self.suffix2 is not None:
            self.suffix2 = torch.tensor(list(map(int, self.suffix2.split(','))), dtype=torch.int32)

        # for example, cut #tokens to 95%ile max tokens to a multiple of 32 for example
        self.cut_percentile = data_config.get('cut_percentile', None)
        self.cut_percentile_multiple = data_config.get('cut_percentile_multiple', None)
        if self.cut_percentile is not None:
            assert isinstance(self.cut_percentile, float) 
            assert self.cut_percentile_multiple is not None
            assert isinstance(self.cut_percentile_multiple, int)

        self.padding_index = data_config.get('padding_index', 0)
        dtype_map = {'uint16': np.uint16, 'uint32': np.uint32}
        self.dtype = dtype_map[data_config.get('dtype', 'uint16')]
        self.target = parse_target(data_config['target'])
        self.populate_all_files(data_config['file_pattern'])
        self.reader_queue_len = data_config.get('reader_queue_len', 2)
        self.num_reader_threads = data_config.get('num_reader_threads', 1)

        dl_config = config['dl']
        self.batch_size = dl_config['batch_size']
        assert self.batch_size % world_size == 0, "batch_size needs to be a multiple of world_size"
        self.batches_queue_len = dl_config.get('batches_queue_len', 100)

        if self.rank == 0:
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
        return np.fromfile(fpath, dtype=self.dtype)

    def _populate_file_contents_queue(self, file_path):
        # check whether pool has been stopped before starting file read
        if not self.stopped:
            file_data = self._read_file(file_path)
            feat1, feat2 = read_pairs(file_data, self.maxlen1, self.maxlen2, self.padding_index)
        # check whether pool has been stopped before putting data into queue
        if not self.stopped:
            # put read data in queue, this waits when queue is full until someone reads from it
            self._file_contents_queue.put((file_path, feat1, feat2))

    def _submit_file_reads_async(self):
        while not self.stopped:
            np.random.shuffle(self.all_files)
            _res = self._file_reader_pool.map(self._populate_file_contents_queue, self.all_files)
            assert all(map(lambda x: x is None, _res))

    def _create_batches_async(self):
        while not self.stopped:
            file_path, feats1, feats2 = self._file_contents_queue.get(timeout=600)
            logger.info(f'creating batches from {file_path} already prepared batches {self.batches_prepared}')
            self._file_contents_queue.task_done()
            if self._data_from_prev_file is not None:
                if self._prev_file != file_path:
                    feats1 = torch.cat((feats1, self._data_from_prev_file[0]), 0)
                    feats2 = torch.cat((feats2, self._data_from_prev_file[1]), 0)
                self._data_from_prev_file = None
                self._prev_file = None
            n = feats1.shape[0]
            order = torch.randperm(n)
            feats1, feats2 = feats1[order], feats2[order]
            feats1_split = torch.split(feats1, self.batch_size, dim=0)
            feats2_split = torch.split(feats2, self.batch_size, dim=0)
            if len(feats1_split[-1]) != self.batch_size:
                self._data_from_prev_file = (feats1_split[-1], feats2_split[-1])
                self._prev_file = file_path
                feats1_split, feats2_split = feats1_split[:-1], feats2_split[:-1]
            for feat_pair in zip(feats1_split, feats2_split):
                if self.stopped:
                    break
                self._train_batches_queue.put(feat_pair)
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

    def input_ids_to_dict_feat(self, feat):
        return {
            'input_ids': feat.long(),
            'attention_mask': (feat!=self.padding_index).long()
        }

    def __iter__(self):
        while self.num_batches < self.num_steps:
            self.num_batches += 1
            feats1, feats2 = [], []
            if self.rank == 0:
                feats1, feats2 = self._train_batches_queue.get(timeout=600)
                self._train_batches_queue.task_done()
                if self.prefix1 is not None:
                    feats1 = torch.cat((self.prefix1[None, :].repeat(self.batch_size, 1), feats1),
                                       dim=1)[:, :self.maxlen1]
                if self.prefix2 is not None:
                    feats2 = torch.cat((self.prefix2[None, :].repeat(self.batch_size, 1), feats2),
                                       dim=1)[:, :self.maxlen2]
                if self.suffix1 is not None:
                    start_cols = torch.minimum((feats1>0).sum(1)+len(self.suffix1), torch.tensor(self.maxlen1))-len(self.suffix1)
                    start_rows = torch.arange(feats1.shape[0])
                    for _i in range(len(self.suffix1)):
                        feats1[start_rows, start_cols+_i] = self.suffix1[_i]
                if self.suffix2 is not None:
                    start_cols = torch.minimum((feats2>0).sum(1)+len(self.suffix2), torch.tensor(self.maxlen2))-len(self.suffix2)
                    start_rows = torch.arange(feats2.shape[0])
                    for _i in range(len(self.suffix2)):
                        feats2[start_rows, start_cols+_i] = self.suffix2[_i]
                feats1 = list(feats1.to(device='cuda').tensor_split(self.world_size))
                feats2 = list(feats2.to(device='cuda').tensor_split(self.world_size))

            feats1_dest = torch.empty((self.batch_size//self.world_size, self.maxlen1), device='cuda',
                                      dtype=torch.int32)
            feats2_dest = torch.empty((self.batch_size//self.world_size, self.maxlen2), device='cuda',
                                      dtype=torch.int32)
            # feats1 and feats2 are now loaded and can be transferred to other ranks
            if self.world_size > 1:
                feats1, handler1 = scatter_tensor_from_one(feats1_dest, feats1)
                feats2, handler2 = scatter_tensor_from_one(feats2_dest, feats2)
                # to atleast overlap communication of both sides of features
                if handler1 is not None:
                    handler1.wait()
                if handler2 is not None:
                    handler2.wait()
            else:
                feats1 = feats1[0]
                feats2 = feats2[0]

            if self.cut_percentile is not None:
                def next_multiple(num, multiple):
                    num += multiple-1
                    num -= num%multiple
                    return num
                percentile_pos = min(int(feats1.shape[0]*self.cut_percentile), feats1.shape[0]-1)
                cut_feat1 = torch.sort((feats1!=self.padding_index).sum(1)).values[percentile_pos]
                feats1 = feats1[:, :next_multiple(cut_feat1, self.cut_percentile_multiple)]
                cut_feat2 = torch.sort((feats2!=self.padding_index).sum(1)).values[percentile_pos]
                feats2 = feats2[:, :next_multiple(cut_feat2, self.cut_percentile_multiple)]
                if self.rank == 0:
                    logger.info(f'{feats1.shape[1]} {feats2.shape[1]}')

            feats1 = self.input_ids_to_dict_feat(feats1)
            feats2 = self.input_ids_to_dict_feat(feats2)
            batch_data = {'feats': {'text1': feats1, 'text2': feats2}}
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


class BinLPairsAdlDataLoader(BinLPairsDataLoader):
    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.adl = connect_adl()
        super(BinLPairsAdlDataLoader, self).__init__(config, rank, world_size, device, *args, **kwargs)

    def populate_all_files(self, file_pattern):
        self.all_files = []
        for pattern in file_pattern.split('|'):
            self.all_files.extend(self.adl.glob(pattern))

    def _read_file(self, fpath):
        with self.adl.open(fpath, 'rb') as f:
            bytesdata = f.read()
        return np.frombuffer(bytesdata, dtype=self.dtype)


class BinLPairsDataLoaderMultitask():
    prepare_fn = lambda *args: None
    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.local_world_size = kwargs.get('local_world_size', self.world_size)
        # num_steps here is sum from individual datasets
        data_config = config['data']
        self.datasets = list(data_config['files'].keys())
        logger.info(f'datasets: {self.datasets}')
        self.adl_enabled = data_config.get('adl', True)
        num_steps_datasets = np.array([data_config['files'][d]['num_steps'] for d in self.datasets], dtype=np.int32)
        self.num_steps, self.num_batches = num_steps_datasets.sum(), 0

        self.order = np.array(sum([[i]*_len for i, _len in enumerate(num_steps_datasets)], []))
        logger.info(f"{np.random.get_state()[1][:5]}")
        np.random.shuffle(self.order)
        logger.info(f"{np.random.get_state()[1][:5]}")

        # if world_size > 1:
        #     orders = all_gather(self.order, world_size)
        #     for i in range(1, world_size):
        #         assert np.array_equal(orders[0], orders[i]), "Shuffle order not same across devices"
        #     logger.info("Shuffle order same across devices")

        self.dls = {}
        for d in self.datasets:
            config_copy = copy.deepcopy(config)
            del config_copy['data']['files']
            config_copy['num_steps'] = data_config['files'][d]['num_steps']
            config_copy['data']['reader_queue_len'] = data_config['files'][d].get('reader_queue_len', 2)
            config_copy['data']['num_reader_threads'] = data_config['files'][d].get('num_reader_threads', 1)
            config_copy['data']['maxlen1'] = data_config['files'][d]['maxlen1']
            config_copy['data']['maxlen2'] = data_config['files'][d]['maxlen2']
            if 'prefix1' in data_config['files'][d]:
                config_copy['data']['prefix1'] = data_config['files'][d]['prefix1']
            if 'prefix2' in data_config['files'][d]:
                config_copy['data']['prefix2'] = data_config['files'][d]['prefix2']
            if 'suffix1' in data_config['files'][d]:
                config_copy['data']['suffix1'] = data_config['files'][d]['suffix1']
            if 'suffix2' in data_config['files'][d]:
                config_copy['data']['suffix2'] = data_config['files'][d]['suffix2']
            config_copy['data']['file_pattern'] = data_config['files'][d]['file_pattern']
            if self.adl_enabled:
                self.dls[d] = BinLPairsAdlDataLoader(config_copy, rank, world_size, device, *args, **kwargs)
            else:
                self.dls[d] = BinLPairsDataLoader(config_copy, rank, world_size, device, *args, **kwargs)

    def __iter__(self):
        self.dl_iters = [iter(self.dls[x]) for x in self.datasets]
        while self.num_batches < self.num_steps:
            dataset_idx = self.order[self.num_batches]
            batch = next(self.dl_iters[dataset_idx])
            batch['dataset_idx'] = dataset_idx
            self.num_batches += 1
            yield batch

    def join(self):
        for dl in self.dls.values():
            dl.join()

    def __len__(self):
        return self.num_steps

    def parse_steps(self, steps):
        assert isinstance(steps, int), "bin lines datareader needs integer steps"
        return steps

    def state_dict(self):
        return {'num_batches': self.num_batches, 'dls': {k: v.state_dict() for k, v in self.dls.items()}}

    def load_state_dict(self, sd):
        self.num_batches = sd['num_batches']
        for k, v in sd['dls']:
            self.dls[k].load_state_dict(v)


