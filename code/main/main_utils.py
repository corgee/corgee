import os
import sys
import yaml
import random
import logging
import datetime
import itertools
import multiprocessing as mp
import numpy as np
import scipy.sparse as sp
import faiss
import hnswlib
import tqdm.autonotebook as tqdm
import torch
import torch.distributed as dist
from sklearn.preprocessing import normalize
from main.xclib.sparse import retain_topk
from itertools import repeat as iter_repeat


logger = logging.getLogger(__name__)


def read_config(config_file, section=None):
    """
    Read the yaml config file
    :param config_file:
    :param section: section in the config file that needs to be read
    :return: A dictionary containing configuration for the section. If section is none, return the whole config
    """
    with open(config_file, "r") as fin:
        try:
            config = yaml.safe_load(os.path.expandvars(fin.read()))
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception
    if section is None:
        return config
    return config[section]


def write_yaml(yaml_data, fname):
    with open(fname, 'w') as fout:
        _ = yaml.dump(yaml_data, fout)


def filter_args_by_start(args, start):
    newargs = []
    chars_to_remove = len(start)
    for i in range(0, len(args), 2):
        if args[i].startswith(start):
            newargs.append(args[i][chars_to_remove:])
            newargs.append(args[i+1])
    return newargs


def replace_keys(_dict, args):
    nested_args = []
    singleton_arg_values = []
    for i in range(0, len(args), 2):
        if '#' in args[i]:
            nested_args.append(args[i].split('#')[0])
        else:
            singleton_arg_values.append((args[i], args[i+1]))
    for nested_arg in nested_args:
        assert nested_arg in _dict, f'nested arg {nested_arg} not in dict'
        subargs = filter_args_by_start(args, nested_arg+'#')
        _dict[nested_arg] = replace_keys(_dict[nested_arg], subargs)
    for sngl_arg, sngl_val in singleton_arg_values:
        assert sngl_arg in _dict, f'singleton arg {sngl_arg} not in dict'
        _dict[sngl_arg] = type(_dict[sngl_arg])(sngl_val)
    return _dict


def override_configs(configs, args):
    """
    Given a dict of configs, parse args to replace some values in the configs
    """
    assert (len(args) % 2) == 0, "number of arguments not even"
    configs = replace_keys(configs, args)
    return configs


def all_gather(item, world_size, filter_none=True):
    if world_size == 1:
        return [item]
    item_list = [None for _ in range(world_size)]
    dist.all_gather_object(item_list, item, group=dist.group.WORLD)
    if filter_none:
        return [x for x in item_list if x is not None]
    else:
        return item_list


def scatter_from_one_local(item_list, rank, local_world_size):
    res_list = [None]
    local_group = dist.new_group(list(range((rank//local_world_size)*local_world_size, (rank//local_world_size+1)*local_world_size)))
    dist.scatter_object_list(res_list, item_list, src=(rank//local_world_size)*local_world_size, group=local_group)
    # cannot be one because src is global rank of source and not group rank
    # dist.scatter_object_list(res_list, item_list, src=0, group=local_group)
    return res_list[0]


def scatter_tensor_from_one(tensor_dest, tensor_list):
    handler = dist.scatter(tensor_dest, scatter_list=tensor_list, async_op=True)
    return tensor_dest, handler


def scatter_from_one(item_list, src=0):
    res_list = [None]
    dist.scatter_object_list(res_list, item_list, src=src, group=dist.group.WORLD)
    return res_list[0]


def all_sync(item, world_size, source=0):
    if world_size == 1:
        return item
    return all_gather(item, world_size)[source]


def all_gather_reduce(item, world_size, reduction='sum'):
    if world_size == 1:
        return item
    elif reduction == 'sum':
        return sum(all_gather(item, world_size))
    elif reduction == 'mean':
        return sum(all_gather(item, world_size))/world_size
    elif reduction == 'dictitems_mean':
        gathered_items = all_gather(item, world_size)
        return {k: np.mean([_item[k] for _item in gathered_items]) for k in gathered_items[0]}
    elif reduction == 'npvstack':
        return np.vstack(all_gather(item, world_size))
    elif reduction == 'dictitems_npvstack':
        gathered_items = all_gather(item, world_size)
        return {k: np.vstack([_item[k] for _item in gathered_items]) for k in gathered_items[0]}
    elif reduction == 'npconcat':
        return np.concatenate(all_gather(item, world_size))
    elif reduction == 'spvstack':
        return sp.vstack(all_gather(item, world_size))
    elif reduction == 'listconcat':
        return list(itertools.chain(*all_gather(item, world_size)))
    elif reduction == 'shape':
        items_list = all_gather(item, world_size)
        print(item, items_list, world_size)
        return (sum([x[0] for x in items_list]), sum([x[1] for x in items_list])//world_size)
    elif reduction == 'weighted_mean':
        weights = all_gather(item[0], world_size)
        items_list = all_gather(item[1], world_size)
        return sum([x*y for x, y in zip(weights, items_list)]) / sum(weights)
    else:
        assert False, f'unknown reduction {reduction}'


def move_to_device(batch, device):
    def move_to_device_(x):
        return move_to_device(x, device)
    if isinstance(batch, dict):
        batch = {k: move_to_device_(v) for k, v in batch.items()}
    elif isinstance(batch, list):
        batch = list(map(move_to_device_, batch))
    elif isinstance(batch, tuple):
        batch = tuple(map(move_to_device_, batch))
    elif isinstance(batch, np.ndarray):
        batch = torch.tensor(batch).to(device, non_blocking=True)
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device, non_blocking=True)
    elif batch is None or isinstance(batch, (str, int, float, bool, np.integer)):
        pass
    else:
        raise NotImplementedError
    return batch


def model_same_across_devices(model, rank: int, world_size: int) -> bool:
    if world_size == 1:
        return True
    sds = all_gather({k: v.cpu() for k, v in model.state_dict().items()}, world_size)
    res = True
    if rank == 0:
        for k, i in itertools.product(sds[0].keys(), range(1, world_size)):
            if not torch.all(torch.eq(sds[0][k], sds[i][k])):
                res = False
                break
    res = all_sync(res, world_size, source=0)
    return res


def split_by_count(batch_data, count):
    if isinstance(batch_data, dict):
        keys = list(batch_data.keys())
        split_batch_values = [split_by_count(batch_data[k], count) for k in keys]
        if all([isinstance(x, iter_repeat) for x in split_batch_values]):
            return iter_repeat(batch_data)
        return [dict(zip(kk, tt)) for kk, tt in zip(iter_repeat(keys), zip(*split_batch_values))]
    elif isinstance(batch_data, list):
        split_batch_data = [split_by_count(x, count) for x in batch_data]
        if all([isinstance(x, iter_repeat) for x in split_batch_data]):
            return iter_repeat(batch_data)
        return list(map(list, zip(*split_batch_data)))
    elif isinstance(batch_data, torch.Tensor):
        return batch_data.tensor_split(count)
    elif isinstance(batch_data, (int, float, str)):
        return iter_repeat(batch_data)
    else:
        raise NotImplementedError


def split_by_max_batch_size(batch_data, chunk_size):
    if isinstance(batch_data, dict):
        keys = list(batch_data.keys())
        split_batch_values = [split_by_max_batch_size(batch_data[k], chunk_size) for k in keys]
        if all([isinstance(x, iter_repeat) for x in split_batch_values]):
            return iter_repeat(batch_data)
        return [dict(zip(kk, tt)) for kk, tt in zip(iter_repeat(keys), zip(*split_batch_values))]
    elif isinstance(batch_data, list):
        split_batch_data = [split_by_max_batch_size(x, chunk_size) for x in batch_data]
        if all([isinstance(x, iter_repeat) for x in split_batch_data]):
            return iter_repeat(batch_data)
        return list(map(list, zip(*split_batch_data)))
    elif isinstance(batch_data, torch.Tensor):
        return batch_data.split(chunk_size)
    elif isinstance(batch_data, (int, float, str)):
        return iter_repeat(batch_data)
    else:
        raise NotImplementedError


def parse_num_threads(num_threads):
    if isinstance(num_threads, int):
        return num_threads
    elif isinstance(num_threads, str) and num_threads.startswith('cpu:'):
        return int(np.ceil(mp.cpu_count()*float(num_threads[len('cpu:'):])))
    else:
        raise NotImplementedError


def exact_nns(query, data, config, K=10):
    """
    Exact search suitable and designed when there are a lot of docs but queries are less
    """
    assert query.shape[1] == data.shape[1], "data and query dimensions don't match"
    if config['space'] == 'cosine':
        query = normalize(query)
        data = normalize(data)
    else:
        assert config['space'] == 'l2'
    torch.cuda.empty_cache()
    op = faiss.GpuMultipleClonerOptions()
    op.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(data.shape[1]), co=op)
    gpu_index.add(data)
    D, I = gpu_index.search(query, K)
    final_shape = (query.shape[0], data.shape[0])
    distances = 1-(D.ravel()/2.0)
    return sp.csr_matrix((distances, (np.arange(I.shape[0]).repeat(I.shape[1]), I.ravel())), shape=final_shape)


def pq_exact_nns(query, data, config, K=10):
    logger.info('pq exact nns')
    print(config)
    """
    Exact search suitable and designed when there are a lot of docs but queries are less
    """
    assert query.shape[1] == data.shape[1], "data and query dimensions don't match"
    if config['space'] == 'cosine':
        query = normalize(query)
        data = normalize(data)
    else:
        assert config['space'] == 'l2'
    torch.cuda.empty_cache()
    op = faiss.GpuMultipleClonerOptions()
    op.shard = True

    index=faiss.index_factory(data.shape[1], config.get("index_factory", "PQ16"))

    pq_cache = config.get('pq_cache')
    if os.path.exists(pq_cache):
        quantizer = faiss.read_ProductQuantizer(pq_cache)
        index.pq = quantizer
        # index.is_trained = quantizer.is_trained
        index.is_trained = True
        logger.info('loaded quantizer')
    else:
        if config['num_trn'] < len(data):
            data_trn = data[np.random.choice(len(data), size=config['num_trn'], replace=False)]
        else:
            data_trn = data
        index.train(data_trn)
        logger.info('trained quantizer')
        faiss.write_ProductQuantizer(index.pq, pq_cache)

    gpu_index = faiss.index_cpu_to_all_gpus(index, co=op)
    gpu_index.add(data)
    logger.info('added items, searching...')
    D, I = gpu_index.search(query, K)
    logger.info('pq search finished')
    final_shape = (query.shape[0], data.shape[0])
    distances = 1-(D.ravel()/2.0)
    return sp.csr_matrix((distances, (np.arange(I.shape[0]).repeat(I.shape[1]), I.ravel())), shape=final_shape)


def nns(query, data, config, K=10):
    if config['algorithm'] == 'hnsw':
        num_threads, query_batch_size = parse_num_threads(config['num_threads']), config['query_batch_size']
        logger.info('Creating HNSW index')
        hnsw = hnswlib.Index(config['space'], data.shape[1])
        hnsw.init_index(data.shape[0], ef_construction=config['ef_construction'],
                        M=config['M'], random_seed=config['random_seed'])
        hnsw.set_ef(config['ef'])
        logger.info(f'Adding {data.shape[0]} items to HNSW')
        hnsw.add_items(data, np.arange(data.shape[0]), num_threads=num_threads)
        res = []
        pbar = tqdm.tqdm(range(0, query.shape[0], query_batch_size), desc='Querying HNSW')
        for i in pbar:
            res.append(hnsw.knn_query(query[i:i+query_batch_size], k=K, num_threads=num_threads))
        _indices = np.vstack([x[0] for x in res]).flatten()
        _data = -1.0*np.vstack([x[1] for x in res]).flatten()
        pred_smat = sp.csr_matrix((_data, _indices, np.arange(0, K*query.shape[0]+1, K)),
                                  shape=(query.shape[0], data.shape[0]))
        return pred_smat
    elif config['algorithm'] == 'exact':
        return exact_nns(query, data, config, K)
    elif config['algorithm'] == 'exact_pq':
        return pq_exact_nns(query, data, config, K)
    else:
        raise NotImplementedError


class TeeLogger(object):
    def __init__(self, fname):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.log = open(fname, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


def check_overlap(a, b):
    return (len(set(a) & set(b)) > 0)
