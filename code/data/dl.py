import logging
from data.mm_data import MMDataLoader, MMEvalDataLoader, SplitMMDataLoader, MultiTaskMMDataLoader
from data.mm_data_with_neg import MMDataLoaderWithNegatives, MultiTaskMMDataLoaderWithNegatives
from data.unsup_data import UnsupDataLoader, SplitUnsupDataLoader
from data.corpus_mae_data import CorpusMAEDataLoader, CorpusSimpleDataLoader
from data.bin_pairs_reader import BinLPairsDataLoader, BinLPairsAdlDataLoader, BinLPairsDataLoaderMultitask
from data.bin_mlm_reader import BinLMLMLoader, BinLMLMAdlDataLoader, BinLMLMAdlDataLoaderMultitask
from data.nomic.bin_pair_with_neg_reader import BinLPairsWithNegativesDataLoaderMultitaskSingleEpoch
from data.e5_instructions.dataloader import E5InstructDataloader


logger = logging.getLogger(__name__)
dataformat_dl_map = {
    "mm": MMDataLoader,
    "mm_neg": MMDataLoaderWithNegatives,
    "unsup": UnsupDataLoader,
    "split_unsup": SplitUnsupDataLoader,
    "mm_eval": MMEvalDataLoader,
    "split_mm": SplitMMDataLoader,
    "mt_mm": MultiTaskMMDataLoader,
    "mt_mm_neg": MultiTaskMMDataLoaderWithNegatives,
    "corpus_mae": CorpusMAEDataLoader,
    "corpus_simple": CorpusSimpleDataLoader,
    "binl_pairs": BinLPairsDataLoader,
    "binl_adl_pairs": BinLPairsAdlDataLoader,
    "binl_pairs_mt": BinLPairsDataLoaderMultitask,
    "binl_adl_pairs_with_negs_mt_1epoch": BinLPairsWithNegativesDataLoaderMultitaskSingleEpoch,
    "binl_mlm_reader": BinLMLMLoader,
    "binl_adl_mlm_reader": BinLMLMAdlDataLoader,
    "binl_adl_mlm_reader_mt": BinLMLMAdlDataLoaderMultitask,
    "e5_instruct_tune": E5InstructDataloader
}


def create_dl(config, rank, world_size, device, local_rank, local_world_size):
    if config['format'] in dataformat_dl_map:
        return dataformat_dl_map[config['format']](config, rank, world_size, device,
                                                   local_rank=local_rank,
                                                   local_world_size=local_world_size)
    else:
        logger.info(f'config format is {config["format"]}')
        raise NotImplementedError


def prepare_all_config_datasets(config):
    for mode in ['eval', 'train']:
        if mode not in config:
            continue
        mode_configs = config[mode]
        if not isinstance(mode_configs, list):
            mode_configs = [mode_configs]
        for _conf in mode_configs:
            if _conf.get('type', None) == 'beir':
                continue
            assert isinstance(_conf, dict)
            assert 'data' in _conf
            assert 'format' in _conf['data']
            assert _conf['data']['format'] in dataformat_dl_map
            dataformat_dl_map[_conf['data']['format']].prepare_fn(_conf['data'])
