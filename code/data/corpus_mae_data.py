import os
import copy
import glob
import logging
import numpy as np
import torch
from data.data_utils import create_feat, prepare_features, parse_dl_len, parse_target, populate_targets


logger = logging.getLogger(__name__)


def prepare_corpus_mae_dataset(config):
    config = config['files']
    dirname, config['fname'] = os.path.split(config['file'])
    prepare_features(config, dirname)
    logger.info("X features prepared")


# def prepare_split_corpus_mae_dataset(config):
#     config_copy = copy.deepcopy(config)
#     all_dirnames = []
#     for pattern in config['files']['dir'].split('|'):
#         all_dirnames.extend(glob.glob(pattern))
#     for dirname_instance in all_dirnames:
#         config_copy['files']['dir'] = dirname_instance
#         prepare_corpus_mae_dataset(config_copy)

def stack_padding(l, cols=None):
    maxlen = max([len(arr) for arr in l])
    if cols is None:
        cols = maxlen
    else:
        assert cols >= maxlen, "#cols should be atleast maxlen"
    return np.array([np.lib.pad(arr, (0, cols - len(arr)), 'constant', constant_values=0) for arr in l])


def process_hints(ids, masked, unmasked, unnorm_probs, max_qs, num_hints, maxlen, avoid_cls_in_hint):
    # hint mask not needed with the current implementation as we always exactly K number of hints
    bsz = len(masked)
    hintids = np.zeros((bsz, max_qs, num_hints), dtype=np.int32)
    hintpos = np.zeros((bsz, max_qs, num_hints), dtype=np.int32)
    targetid = np.zeros((bsz, max_qs), dtype=np.int32)
    targetpos = np.zeros((bsz, max_qs), dtype=np.int32)
    targetmask = np.zeros((bsz, max_qs), dtype=np.int32)

    for rownum, row_pos_masked in enumerate(masked):
        for hnum, m in enumerate(row_pos_masked):
            unmasked_row = unmasked[rownum]
            if avoid_cls_in_hint:
                unmasked_row = unmasked_row[1:]
            _p = unnorm_probs[unmasked_row-m+maxlen]
            _pos = np.sort(np.random.choice(unmasked_row, size=num_hints, replace=False, p=_p/_p.sum()))
            hintids[rownum, hnum] = ids[rownum][_pos]
            hintpos[rownum, hnum] = _pos
            targetid[rownum, hnum] = ids[rownum][m]
            targetpos[rownum, hnum] = m
            targetmask[rownum, hnum] = 1

    return hintids, hintpos, targetid, targetpos, targetmask


def mask_generate_hints(ids, attmask, unnorm_probs, maxlen, num_hints, mask_rate, min_mask=1, avoid_cls_in_hint=False):
    lens = attmask.sum(1)
    # mask such that atleast 1 token is masked and atleast #hint+1 tokens are unmasked
    masklens = np.clip(np.rint(lens*mask_rate).astype(int), a_min=min_mask, a_max=lens-num_hints-1)

    # sampling random mask and hints
    if avoid_cls_in_hint:
        # sample only starting from 1
        permutations = [1+np.random.permutation(_len-1) for _len in lens]
        masked = [permutations[i][:_masklen] for i, _masklen in enumerate(masklens)]
        # prepend 0 to the start of the unmasked array as CLS is always unmasked
        unmasked = [np.hstack((0,np.sort(permutations[i][_masklen:]))) for i, _masklen in enumerate(masklens)]
    else:
        permutations = [np.random.permutation(_len) for _len in lens]
        masked = [permutations[i][:_masklen] for i, _masklen in enumerate(masklens)]
        unmasked = [np.sort(permutations[i][_masklen:]) for i, _masklen in enumerate(masklens)]

    # # aligning with 4 just in case
    # max_qs = (max(len(x) for x in masked)+3)//4*4
    max_qs = max(len(x) for x in masked)
    # TODO: optimize this
    # this function call reduces from 2000 it/s to 240 it/s
    hintids, hintpos, targetid, targetpos, targetmask = process_hints(ids, masked, unmasked, unnorm_probs, max_qs, num_hints, maxlen, avoid_cls_in_hint)
    # hintids, hintpos, targetid, targetpos, targetmask = None, None, None, None, None
    hints_final = dict(hint_input_ids=hintids, hint_pos=hintpos, target_input_ids=targetid, target_pos=targetpos, target_mask=targetmask)

    unmasked_pos = stack_padding(unmasked)
    unmasked_ids = stack_padding([_ids[_unmasked] for _unmasked, _ids in zip(unmasked, ids)])
    unmasked_attmask = (unmasked_ids>0).astype(int)
    unmasked_final = dict(input_ids=unmasked_ids, attention_mask=unmasked_attmask, pos=unmasked_pos)
    return hints_final, unmasked_final


class CorpusMAEDataset():
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        dirname, config['fname'] = os.path.split(config['file'])
        self.feat = create_feat(config, dirname)

    def load_data(self):
        self.feat.load_data()

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.feat)


class CorpusMAECollator():
    def __init__(self, dataset, config):
        self.config = config
        self.target = parse_target(self.config['target'])
        self.group_feat_key = config['group_feat_key']

        logger.info(f"target: {self.target}, type: {type(self.target)}")
        self.dataset = dataset
        self.data_loaded = False

        self.maxlen = config['maxlen']
        self.num_hints = config['num_hints']
        self.mask_rate = config['mask_rate']
        self.avoid_cls_in_hint = config.get('avoid_cls_in_hint', True)
        self.unnorm_probs = np.exp(-np.square((np.arange(2*self.maxlen+1)-self.maxlen)/config['std_dist_hints']))

    def __call__(self, batch):
        if not self.data_loaded:
            self.dataset.load_data()
            self.all_ids = self.dataset.feat.get_ids()
            self.data_loaded = True
        batch_size = len(batch)
        ids = np.array(batch)
        batch_data = {'batch_size': batch_size, 'feats': {}}

        feats = self.dataset.feat.get_fts(self.all_ids[ids])
        hints, inputs = mask_generate_hints(feats['input_ids'], feats['attention_mask'], self.unnorm_probs,
                                            maxlen=self.maxlen,
                                            num_hints=self.num_hints,
                                            mask_rate=self.mask_rate,
                                            avoid_cls_in_hint=self.avoid_cls_in_hint)
        batch_data['feats'][self.group_feat_key] = {'xfeat': inputs, 'yfeat': hints}
        batch_data['target'] = self.target
        # populate targets is [most likely] used if we need ids for deduping etc while computing loss
        # populate_targets(self.target, None, None)

        return batch_data


class CorpusMAEDataLoader():
    prepare_fn = prepare_corpus_mae_dataset

    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.dl_config = config['dl']
        self.data = CorpusMAEDataset(config['files'], rank, world_size)

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
            collate_fn=CorpusMAECollator(self.data, config),
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


# class SplitCorpusMAEDataLoader(CorpusMAEDataLoader):
#     prepare_fn = prepare_split_corpus_mae_dataset

#     def __init__(self, config, rank, world_size, device, *args, **kwargs):
#         self.config = config
#         self.rank = rank
#         self.world_size = world_size
#         self.device = device

#         self.dl_config = config['dl']
#         all_dirnames = []
#         for pattern in config['files']['dir'].split('|'):
#             all_dirnames.extend(glob.glob(pattern))
#         assert len(all_dirnames) >= world_size, "lesser splits than number of GPU workers"

#         splits = np.array_split(np.arange(len(all_dirnames)), world_size)
#         self.dirnames = [all_dirnames[i] for i in splits[rank]]
#         logger.info(f"Directory names to load from: {';'.join(self.dirnames)}")
#         self.dls = []
#         for dirname in self.dirnames:
#             config_copy = copy.deepcopy(config)
#             config_copy['files']['dir'] = dirname
#             config_copy['force_no_sampler'] = True
#             self.dls.append(CorpusMAEDataLoader(config_copy, rank, world_size, device))

#         self.order_dls = []
#         self.epoch_num, self.num_batches = 0, 0
#         self.len = self.parse_steps(self.config.get('num_steps', 'e1'))

#         # If dataset size is same across two processes, same indices are sampled for both in every epoch leading to same
#         # datapoints going together always. This line would make the rng state different across all processes
#         _ = torch.randperm(1+self.rank)

#     def parse_steps(self, steps):
#         return parse_dl_len(steps, sum([len(_dl.dl) for _dl in self.dls]), self.world_size)

#     def next_dl_iter(self):
#         if len(self.order_dls) == 0:
#             self.order_dls = np.random.permutation(len(self.dls)).tolist()
#             self.epoch_num += 1
#         dl_idx = self.order_dls.pop(0)
#         logger.info(f"Next dataset split: {self.dirnames[dl_idx]}")
#         return self.dls[dl_idx]

#     def __iter__(self):
#         self.current_dl_iter = self.next_dl_iter()
#         while True:
#             if self.num_batches >= self.len:
#                 break
#             # Iterating over the underlying dataloader, bypassing current_dl_iter's __iter__ to loop only once
#             # This works without updating current_dl_iter's num_epochs because we are not using its sampler
#             for batch in self.current_dl_iter.dl:
#                 yield batch
#                 self.num_batches += 1
#                 if self.num_batches >= self.len:
#                     break
#             self.current_dl_iter = self.next_dl_iter()

#     def state_dict(self):
#         return {
#             'epoch_num': self.epoch_num,
#             'num_batches': self.num_batches,
#             'order_dls': self.order_dls
#         }

#     def load_state_dict(self, sd):
#         self.epoch_num = sd['epoch_num']
#         self.num_batches = sd['num_batches']
#         self.order_dls = sd['order_dls']



class CorpusSimpleDataset():
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        dirname, config['fname'] = os.path.split(config['file'])
        self.feat = create_feat(config, dirname)

    def load_data(self):
        self.feat.load_data()

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.feat)


class CorpusSimpleCollator():
    def __init__(self, dataset, config):
        self.config = config
        self.target = parse_target(self.config['target'])
        self.group_feat_key = config['group_feat_key']

        logger.info(f"target: {self.target}, type: {type(self.target)}")
        self.dataset = dataset
        self.data_loaded = False

    def __call__(self, batch):
        if not self.data_loaded:
            self.dataset.load_data()
            self.all_ids = self.dataset.feat.get_ids()
            self.data_loaded = True
        batch_size = len(batch)
        ids = np.array(batch)
        batch_data = {'batch_size': batch_size, 'feats': {}}

        feats = self.dataset.feat.get_fts(self.all_ids[ids])
        batch_data['feats'][self.group_feat_key] = {'yfeat': feats}
        batch_data['target'] = self.target
        # populate targets is [most likely] used if we need ids for deduping etc while computing loss
        # populate_targets(self.target, None, None)
        return batch_data


class CorpusSimpleDataLoader():
    prepare_fn = prepare_corpus_mae_dataset

    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.dl_config = config['dl']
        self.data = CorpusSimpleDataset(config['files'], rank, world_size)

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
            collate_fn=CorpusSimpleCollator(self.data, config),
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
