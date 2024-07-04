import os
import copy
import glob
import logging
import numpy as np
from numba import njit
import scipy.sparse as sp
import torch
from data.data_utils import csrMemmap, check_done, prepare_spmat, create_feat, XYMat, prepare_features, parse_dl_len, \
    parse_target, populate_targets
from main.main_utils import all_gather


logger = logging.getLogger(__name__)


def prepare_mm_with_negatives_dataset(config):
    config = config['files']
    dirname = config['dir']
    prepare_features(config['xfeat'], os.path.join(dirname, 'xfeat'))
    logger.info("X features prepared")
    prepare_features(config['yfeat'], os.path.join(dirname, 'yfeat'))
    logger.info("Y features prepared")
    prepare_spmat(os.path.join(dirname, 'X_Y.txt'), preprocess=config.get('xy_preprocess', None))
    logger.info("XY matrix prepared")
    negatives_file = config['negatives']['file']
    if negatives_file.endswith('.txt'):
        prepare_spmat(negatives_file, preprocess=config.get('xy_preprocess', None))
        logger.info("XY negatives prepared")
    else:
        csrMemmap.verify_path(negatives_file)
        logger.info("XY negatives path verified")


def prepare_mt_mm_with_negatives_dataset(config):
    for files_config in config['files'].values():
        prepare_mm_with_negatives_dataset({'files': files_config})


class MMDatasetWithNegatives():
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        self.inmemory_smats = config.get('inmemory_smats', False)

        self.xfeat = create_feat(config['xfeat'], os.path.join(config['dir'], 'xfeat'),
                                 prefix=config.get('prefix1', None),
                                 suffix=config.get('suffix1', None),
                                 round_tokens=config.get('round_tokens', None))
        self.yfeat = create_feat(config['yfeat'], os.path.join(config['dir'], 'yfeat'),
                                 prefix=config.get('prefix2', None),
                                 suffix=config.get('suffix2', None),
                                 round_tokens=config.get('round_tokens', None))
        self.xy_smat = XYMat(os.path.join(config['dir'], 'X_Y.txt'), 
                             inmemory=self.inmemory_smats,
                             preprocess=config.get('xy_preprocess', None))
        if config['negatives']['file'].startswith('.txt'):
            raise NotImplementedError
        else:
            self.xy_smat_neg = None

    def load_data(self):
        self.xfeat.load_data()
        self.yfeat.load_data()
        self.xy_smat.load_data()
        neg_file = self.config['negatives']['file']
        if neg_file.startswith('.txt'):
            raise NotImplementedError
        else:
            if self.inmemory_smats:
                self.xy_smat_neg = csrMemmap.load(neg_file)
            else:
                self.xy_smat_neg = csrMemmap(neg_file)
                

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.xy_smat)


def sort_prune_csr(matrix, k=None):
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i+1]
        order = matrix.data[start:end].argsort()[::-1]
        matrix.data[start:end] = matrix.data[start:end][order]
        matrix.indices[start:end] = matrix.indices[start:end][order]
        if k is not None:
            matrix.data[start:end][:k] = 0
    matrix.eliminate_zeros()


@njit
def sort_csr(ind, data, indptr):
    bsz = len(indptr)-1
    for i in range(bsz):
        start, end = indptr[i], indptr[i+1]
        order = data[start:end].argsort()[::-1]
        data[start:end] = data[start:end][order]
        ind[start:end] = ind[start:end][order]

@njit
def sort_ind_csr(ind, indptr):
    bsz = len(indptr)-1
    for i in range(bsz):
        ind[indptr[i]:indptr[i+1]].sort()

@njit
def get_pos_neg(res, y_ind, y_indptr, yneg_ind, yneg_data, yneg_indptr, neg_start, neg_end, neg_num, rand):
    sort_csr(yneg_ind, yneg_data, yneg_indptr)
    sort_ind_csr(y_ind, y_indptr)
    bsz = len(y_indptr)-1
    for i in range(bsz):
        pos_row = y_ind[y_indptr[i]:y_indptr[i+1]]
        # sample the positive
        res[i, 0] = pos_row[int(len(pos_row)*rand[i, 0])]
        neg_row = yneg_ind[yneg_indptr[i]:yneg_indptr[i+1]][neg_start:neg_end]
        # filter positives from negatives
        valid_idx = []
        for j in range(len(neg_row)):
            if np.searchsorted(pos_row, neg_row[j]) == neg_row[j]:
                continue
            valid_idx.append(j)
        # sample negatives
        for j in range(neg_num):
            if len(valid_idx) == 0:
                break
            idx = int(len(valid_idx)*rand[i, j])
            res[i, 1+j] = neg_row[idx]
            valid_idx.pop(idx)


class MMCollatorWithNegatives():
    def __init__(self, dataset, config):
        self.config = config
        self.group_feat_key = config.get('group_feat_key', None)
        self.target = parse_target(self.config['target'])

        logger.info(f"target: {self.target}, type: {type(self.target)}")
        self.dataset = dataset
        self.data_loaded = False

        # negative sampling parameters
        self.yids = None
        self.neg_start = config['files']['negatives']['start']
        self.neg_end = config['files']['negatives']['end']
        self.neg_num = config['files']['negatives']['num']

    def __call__(self, batch):
        if not self.data_loaded:
            self.dataset.load_data()
            # if isinstance(self.dataset.xy_smat_neg, csrMemmap):
            #     sort_prune_csr(self.smat.xy_smat_neg, k=self.neg_end)
            # else:
            #     sort_prune_csr(self.xy_smat_neg, k=self.neg_end)
            self.yids = self.dataset.yfeat.get_ids()
            self.data_loaded = True
        batch_size = len(batch)
        ids = np.array(batch)
        y = self.dataset.xy_smat.smat[ids]
        batch_data = {'batch_size': batch_size, 'feats': {}}
        batch_x_ids = self.dataset.xy_smat.ids[ids]
        xfeats = self.dataset.xfeat.get_fts(batch_x_ids)

        yneg = self.dataset.xy_smat_neg[ids]
        batch_y_inds = np.zeros((batch_size, 1+self.neg_num), dtype=np.int32)-1
        get_pos_neg(batch_y_inds, y.indices, y.indptr, yneg.indices, yneg.data, yneg.indptr,
                                        self.neg_start, self.neg_end, self.neg_num,
                                        np.random.uniform(size=(batch_size,1+self.neg_num)))
        # TODO: add masking before training on very dense datasets. are there any?
        # pos_neg_mask = (batch_y_inds >= 0)
        yfeats = self.dataset.yfeat.get_fts(batch_y_inds)
        if self.group_feat_key is None:
            batch_data['feats'].update(xfeats)
            batch_data['feats'].update(yfeats)
        else:
            # group feats into single instance for ex for sequential execution
            batch_data['feats'] = {self.group_feat_key: {'xfeat': xfeats, 'yfeat': yfeats}}

        batch_data['target'] = self.target
        # expand targets if needed with xids and yids
        # populate_targets(self.target, batch_x_ids, batch_y_inds)

        return batch_data


class MMDataLoaderWithNegatives():
    prepare_fn = prepare_mm_with_negatives_dataset

    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.dl_config = config['dl']
        self.data = MMDatasetWithNegatives(config['files'], rank, world_size)

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
            collate_fn=MMCollatorWithNegatives(self.data, config),
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


class MultiTaskMMDataLoaderWithNegatives(MMDataLoaderWithNegatives):
    prepare_fn = prepare_mt_mm_with_negatives_dataset

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
            self.dls.append(MMDataLoaderWithNegatives(config_copy, rank, world_size, device))

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
