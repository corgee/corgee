import copy
import logging
import numpy as np
from main.main_utils import split_by_count, scatter_from_one
from data.bin_pairs_with_negs_reader import BinLPairsWithNegativesAdlDataLoader, BinLPairsWithNegativesDataLoader


logger = logging.getLogger(__name__)


class BinLPairsWithNegativesDataLoaderMultitaskSingleEpoch():
    prepare_fn = lambda *args: None
    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        # num_steps here is sum from individual datasets
        data_config = config['data']
        self.datasets = list(data_config['files'].keys())
        logger.info(f'datasets: {self.datasets}')
        self.adl_enabled = data_config.get('adl', True)
        num_steps_datasets = np.array([data_config['files'][d]['num_steps'] for d in self.datasets], dtype=np.int32)
        self.num_steps, self.num_batches = num_steps_datasets.sum(), 0
        if 'order' in data_config:
            logger.info(f'loading order from {data_config["order"]}')
            self.order = np.load(data_config["order"])
            assert len(self.order) == self.num_steps, "order file of incorrect length"
            for i, lendataset in enumerate(num_steps_datasets):
                assert (self.order==i).sum() == lendataset, "order num batches from dataset {i} of incorrect length"
        else:
            self.order = np.array(sum([[i]*_len for i, _len in enumerate(num_steps_datasets)], []))
            logger.info(f"{np.random.get_state()[1][:5]}")
            np.random.shuffle(self.order)
            logger.info(f"{np.random.get_state()[1][:5]}")

        self.dls = {}
        for d in self.datasets:
            config_copy = copy.deepcopy(config)
            del config_copy['data']['files']
            config_copy['num_steps'] = data_config['files'][d]['num_steps']
            config_copy['data']['reader_queue_len'] = data_config['files'][d].get('reader_queue_len', 2)
            config_copy['data']['num_reader_threads'] = data_config['files'][d].get('num_reader_threads', 1)
            config_copy['data']['maxlen1'] = data_config['files'][d]['maxlen1']
            config_copy['data']['maxlen2'] = data_config['files'][d]['maxlen2']
            config_copy['data']['num_negatives'] = data_config['files'][d]['num_negatives']
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
                self.dls[d] = BinLPairsWithNegativesAdlDataLoader(config_copy, rank, world_size, device, *args, **kwargs)
            else:
                self.dls[d] = BinLPairsWithNegativesDataLoader(config_copy, rank, world_size, device, *args, **kwargs)

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
