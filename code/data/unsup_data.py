import copy
import glob
import logging
import numpy as np
import torch
from data.data_utils import create_feat, prepare_features, parse_dl_len, parse_target, populate_targets


logger = logging.getLogger(__name__)


def prepare_unsup_dataset(config):
    config = config['files']
    dirname = config['dir']
    prepare_features(config['feat'], dirname)
    logger.info("X features prepared")


def prepare_split_unsup_dataset(config):
    config_copy = copy.deepcopy(config)
    all_dirnames = []
    for pattern in config['files']['dir'].split('|'):
        all_dirnames.extend(glob.glob(pattern))
    for dirname_instance in all_dirnames:
        config_copy['files']['dir'] = dirname_instance
        prepare_unsup_dataset(config_copy)


class UnsupDataset():
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        self.feat = create_feat(config['feat'], config['dir'])

    def load_data(self):
        self.feat.load_data()

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.feat)


class UnsupCollator():
    def __init__(self, dataset, config):
        self.config = config
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
        batch_data = {'batch_size': batch_size, 'feats': {}}
        batch_data['feats'].update(self.dataset.feat.get_fts(ids))
        batch_data['target'] = self.target
        # populate targets is [most likely] used if we need ids for deduping etc while computing loss
        populate_targets(self.target, ids, ids)

        return batch_data


class UnsupDataLoader():
    prepare_fn = prepare_unsup_dataset

    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.dl_config = config['dl']
        self.data = UnsupDataset(config['files'], rank, world_size)

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
            collate_fn=UnsupCollator(self.data, config),
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


class SplitUnsupDataLoader(UnsupDataLoader):
    prepare_fn = prepare_split_unsup_dataset

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
            self.dls.append(UnsupDataLoader(config_copy, rank, world_size, device))

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
