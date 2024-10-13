import logging
from data.bin_pairs_reader import BinLPairsDataLoaderMultitask
from data.bin_pairs_with_negs_reader import BinLPairsWithNegativesDataLoaderMultitask


logger = logging.getLogger(__name__)
dataformat_dl_map = {
    "binl_pairs_mt": BinLPairsDataLoaderMultitask,
    "binl_pairs_wnegs_mt": BinLPairsWithNegativesDataLoaderMultitask,
}


def create_dl(config, rank, world_size, device, local_rank, local_world_size):
    if config['format'] in dataformat_dl_map:
        return dataformat_dl_map[config['format']](config, rank, world_size, device,
                                                   local_rank=local_rank,
                                                   local_world_size=local_world_size)
    else:
        logger.info(f'config format is {config["format"]}')
        raise NotImplementedError
