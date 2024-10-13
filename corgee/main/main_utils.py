import logging
import os
import sys
from itertools import repeat as iter_repeat

import numpy as np
import torch
import torch.distributed as dist
import yaml

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


def filter_args_by_start(args, start):
    newargs = []
    chars_to_remove = len(start)
    for i in range(0, len(args), 2):
        if args[i].startswith(start):
            newargs.append(args[i][chars_to_remove:])
            newargs.append(args[i + 1])
    return newargs


def replace_keys(_dict, args):
    nested_args = []
    singleton_arg_values = []
    for i in range(0, len(args), 2):
        if "#" in args[i]:
            nested_args.append(args[i].split("#")[0])
        else:
            singleton_arg_values.append((args[i], args[i + 1]))
    for nested_arg in nested_args:
        assert nested_arg in _dict, f"nested arg {nested_arg} not in dict"
        subargs = filter_args_by_start(args, nested_arg + "#")
        _dict[nested_arg] = replace_keys(_dict[nested_arg], subargs)
    for sngl_arg, sngl_val in singleton_arg_values:
        assert sngl_arg in _dict, f"singleton arg {sngl_arg} not in dict"
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


def scatter_tensor_from_one(tensor_dest, tensor_list):
    handler = dist.scatter(tensor_dest, scatter_list=tensor_list, async_op=True)
    return tensor_dest, handler


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


def split_by_max_batch_size(batch_data, chunk_size):
    if isinstance(batch_data, dict):
        keys = list(batch_data.keys())
        split_batch_values = [
            split_by_max_batch_size(batch_data[k], chunk_size) for k in keys
        ]
        if all([isinstance(x, iter_repeat) for x in split_batch_values]):
            return iter_repeat(batch_data)
        return [
            dict(zip(kk, tt))
            for kk, tt in zip(iter_repeat(keys), zip(*split_batch_values))
        ]
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
    return len(set(a) & set(b)) > 0
