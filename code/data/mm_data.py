import os
import copy
import glob
import logging
import numpy as np
import scipy.sparse as sp
import torch
from data.data_utils import csrMemmap, check_done, prepare_spmat, create_feat, XYMat, prepare_features, parse_dl_len, \
    parse_target, populate_targets
from main.main_utils import all_gather


logger = logging.getLogger(__name__)


def prepare_mm_dataset(config):
    config = config['files']
    dirname = config['dir']
    prepare_features(config['xfeat'], os.path.join(dirname, 'xfeat'))
    logger.info("X features prepared")
    prepare_features(config['yfeat'], os.path.join(dirname, 'yfeat'))
    logger.info("Y features prepared")
    prepare_spmat(os.path.join(dirname, 'X_Y.txt'), preprocess=config.get('xy_preprocess', None))
    logger.info("XY matrix prepared")


def prepare_split_mm_dataset(config):
    config_copy = copy.deepcopy(config)
    all_dirnames = []
    for pattern in config['files']['dir'].split('|'):
        all_dirnames.extend(glob.glob(pattern))
    for dirname_instance in all_dirnames:
        config_copy['files']['dir'] = dirname_instance
        prepare_mm_dataset(config_copy)


def prepare_mt_mm_dataset(config):
    for files_config in config['files'].values():
        prepare_mm_dataset({'files': files_config})


class MMDataset():
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        self.xfeat = create_feat(config['xfeat'], os.path.join(config['dir'], 'xfeat'), padding_idx=config.get('padding_idx', 0))
        self.yfeat = create_feat(config['yfeat'], os.path.join(config['dir'], 'yfeat'), padding_idx=config.get('padding_idx', 0))
        self.xy_smat = XYMat(os.path.join(config['dir'], 'X_Y.txt'), preprocess=config.get('xy_preprocess', None))

    def load_data(self):
        self.xfeat.load_data()
        self.yfeat.load_data()
        self.xy_smat.load_data()

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.xy_smat)


class MMCollator():
    def __init__(self, dataset, config):
        self.config = config
        self.group_feat_key = config.get('group_feat_key', None)
        self.target = parse_target(self.config['target'])

        logger.info(f"target: {self.target}, type: {type(self.target)}")
        self.dataset = dataset
        self.data_loaded = False

    def __call__(self, batch):
        if not self.data_loaded:
            self.dataset.load_data()
            self.data_loaded = True
        batch_size = len(batch)
        ids = np.array(batch)
        y = self.dataset.xy_smat.smat[ids]
        batch_data = {'batch_size': batch_size, 'feats': {}}
        batch_x_ids = self.dataset.xy_smat.ids[ids]
        xfeats = self.dataset.xfeat.get_fts(batch_x_ids)
        # # TODO; handle this better, temporary fix
        # batch_y_inds = np.array([np.random.choice(y[i].indices, p=y[i].data) for i in range(y.shape[0])])
        batch_y_inds = np.array([np.random.choice(y[i].indices) for i in range(y.shape[0])])
        yfeats = self.dataset.yfeat.get_fts(batch_y_inds)

        if self.group_feat_key is None:
            batch_data['feats'].update(xfeats)
            batch_data['feats'].update(yfeats)
        else:
            # group feats into single instance for ex for sequential execution
            batch_data['feats'] = {self.group_feat_key: {'xfeat': xfeats, 'yfeat': yfeats}}

        batch_data['target'] = self.target
        # expand targets if needed with xids and yids
        populate_targets(self.target, batch_x_ids, batch_y_inds)

        return batch_data


class MMDataLoader():
    prepare_fn = prepare_mm_dataset

    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.dl_config = config['dl']
        self.data = MMDataset(config['files'], rank, world_size)

        if world_size > 1 and (not config.get('force_no_sampler', False)):
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                self.data,
                num_replicas=world_size,
                rank=rank,
                shuffle=self.dl_config['shuffle']
            )
        else:
            self.sampler = None
        assert self.dl_config['batch_size'] % world_size == 0, "batch_size needs to be a multiple of world_size"
        dl_shuffle = False if self.sampler else self.dl_config['shuffle']
        self.dl = torch.utils.data.DataLoader(
            self.data,
            batch_size=self.dl_config['batch_size']//world_size,
            num_workers=self.dl_config['num_workers'],
            collate_fn=MMCollator(self.data, config),
            shuffle=dl_shuffle,
            pin_memory=False,
            sampler=self.sampler,
            drop_last=self.dl_config['drop_last'],
            prefetch_factor=self.dl_config.get('prefetch_factor', 5 if self.dl_config['num_workers'] > 0 else 2)
        )
        self.epoch_num, self.num_batches = 0, 0
        self.len = self.parse_steps(self.config.get('num_steps', 'e1'))

    def parse_steps(self, steps):
        return parse_dl_len(steps, len(self.dl), self.world_size)

    def reset(self):
        self.epoch_num, self.num_batches = 0, 0

    def __iter__(self):
        while True:
            if self.num_batches >= self.len:
                break
            if self.sampler:
                self.sampler.set_epoch(self.epoch_num)
            for batch in self.dl:
                self.num_batches += 1
                if self.num_batches > self.len:
                    break
                yield batch
            self.epoch_num += 1

    def __len__(self):
        return self.len

    def state_dict(self):
        return {
            'epoch_num': self.epoch_num,
            'num_batches': self.num_batches
        }

    def load_state_dict(self, sd):
        self.epoch_num = sd['epoch_num']
        self.num_batches = sd['num_batches']


class SplitMMDataLoader(MMDataLoader):
    prepare_fn = prepare_split_mm_dataset

    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.dl_config = config['dl']
        all_dirnames = []
        for pattern in config['files']['dir'].split('|'):
            all_dirnames.extend(glob.glob(pattern))
        assert len(all_dirnames) >= world_size, "lesser splits than number of GPU workers"

        splits = np.array_split(np.arange(len(all_dirnames)), world_size)
        self.dirnames = [all_dirnames[i] for i in splits[rank]]
        logger.info(f"Directory names to load from: {';'.join(self.dirnames)}")
        self.dls = []
        for dirname in self.dirnames:
            config_copy = copy.deepcopy(config)
            config_copy['files']['dir'] = dirname
            config_copy['force_no_sampler'] = True
            self.dls.append(MMDataLoader(config_copy, rank, world_size, device))

        self.order_dls = []
        self.epoch_num, self.num_batches = 0, 0
        self.len = self.parse_steps(self.config.get('num_steps', 'e1'))

        # If dataset size is same across two processes, same indices are sampled for both in every epoch leading to same
        # datapoints going together always. This line would make the rng state different across all processes
        _ = torch.randperm(1+self.rank)

    def parse_steps(self, steps):
        return parse_dl_len(steps, sum([len(_dl.dl) for _dl in self.dls]), self.world_size)

    def next_dl_iter(self):
        if len(self.order_dls) == 0:
            self.order_dls = np.random.permutation(len(self.dls)).tolist()
            self.epoch_num += 1
        dl_idx = self.order_dls.pop(0)
        logger.info(f"Next dataset split: {self.dirnames[dl_idx]}")
        return self.dls[dl_idx]

    def __iter__(self):
        self.current_dl_iter = self.next_dl_iter()
        while True:
            if self.num_batches >= self.len:
                break
            # Iterating over the underlying dataloader, bypassing current_dl_iter's __iter__ to loop only once
            # This works without updating current_dl_iter's num_epochs because we are not using its sampler
            for batch in self.current_dl_iter.dl:
                yield batch
                self.num_batches += 1
                if self.num_batches >= self.len:
                    break
            self.current_dl_iter = self.next_dl_iter()

    def state_dict(self):
        return {
            'epoch_num': self.epoch_num,
            'num_batches': self.num_batches,
            'order_dls': self.order_dls
        }

    def load_state_dict(self, sd):
        self.epoch_num = sd['epoch_num']
        self.num_batches = sd['num_batches']
        self.order_dls = sd['order_dls']


class MultiTaskMMDataLoader(MMDataLoader):
    prepare_fn = prepare_mt_mm_dataset

    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.dl_config = config['dl']

        self.dls = []
        for key in config['files']:
            config_copy = copy.deepcopy(config)
            config_copy['files'] = config['files'][key]
            self.dls.append(MMDataLoader(config_copy, rank, world_size, device))

        self.epoch_num, self.num_batches = 0, 0
        self.len = sum([len(dl) for dl in self.dls])
        self.order = np.array(sum([[i]*len(dl) for i, dl in enumerate(self.dls)], []))
        np.random.shuffle(self.order)
        if world_size > 1:
            orders = all_gather(self.order, world_size)
            for i in range(1, world_size):
                assert np.array_equal(orders[0], orders[i]), "Shuffle order not same across devices"

    def parse_steps(self, steps):
        return sum([dl.parse_steps(steps) for dl in self.dls])

    def __iter__(self):
        self.dl_iters = [iter(dl) for dl in self.dls]
        for idx in self.order:
            yield next(self.dl_iters[idx])
            self.num_batches += 1

    def state_dict(self):
        return {
            'epoch_num': self.epoch_num,
            'num_batches': self.num_batches,
            'order': self.order
        }

    def load_state_dict(self, sd):
        self.epoch_num = sd['epoch_num']
        self.num_batches = sd['num_batches']
        self.order = sd['order']


class MMEvalCollator():
    def __init__(self, rank, world_size, feat):
        self.rank = rank
        self.world_size = world_size
        self.feat = feat
        self.data_loaded = False

    def __call__(self, batch):
        if not self.data_loaded:
            self.feat.load_data()
            self.data_loaded = True
        batch = np.array(batch)
        indices = np.array_split(batch, self.world_size)[self.rank]
        if len(indices):
            return self.feat.get_fts(indices)
        else:
            return None


class MMEvalDataLoader():
    prepare_fn = prepare_mm_dataset

    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        config_files = config['files']
        self.dl_config = config['dl']

        xy_data_dir = os.path.join(config_files['dir'], 'X_Y.txt')+'.processed'
        check_done(xy_data_dir)
        self.xy_ids = np.load(os.path.join(xy_data_dir, 'ids.npy'))
        if self.rank == 0:
            self.xy_smat = csrMemmap.load(os.path.join(xy_data_dir, 'smat'))
            self.xy_filter = None
            filter_file = os.path.join(config_files['dir'], 'X_Y_overlap.npz')
            if os.path.exists(filter_file):
                self.xy_filter = sp.load_npz(filter_file)

        self.xfeat = create_feat(config_files['xfeat'], os.path.join(config_files['dir'], 'xfeat'), padding_idx=config.get('padding_idx', 0))
        assert self.dl_config['batch_size'] % world_size == 0, "batch_size needs to be a multiple of world_size"
        self.xdl = torch.utils.data.DataLoader(
            self.xy_ids,
            batch_size=self.dl_config['batch_size'],
            num_workers=self.dl_config['num_workers'],
            collate_fn=MMEvalCollator(self.rank, self.world_size, self.xfeat),
            prefetch_factor=self.dl_config.get('prefetch_factor', 5 if self.dl_config['num_workers'] > 0 else None)
        )

        self.yfeat = create_feat(config_files['yfeat'], os.path.join(config_files['dir'], 'yfeat'), padding_idx=config.get('padding_idx', 0))
        self.ydl = torch.utils.data.DataLoader(
            self.yfeat.get_ids(),
            batch_size=self.dl_config['batch_size'],
            num_workers=self.dl_config['num_workers'],
            collate_fn=MMEvalCollator(self.rank, self.world_size, self.yfeat),
            prefetch_factor=self.dl_config.get('prefetch_factor', 5 if self.dl_config['num_workers'] > 0 else None)
        )
