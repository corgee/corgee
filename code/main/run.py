'''
Run this with 
    torchrun --standalone --nnodes=1 --nproc-per-node=2 code/main/run.py configs/<config_path>.yaml
'''

import os
import sys
import git
import logging
import torch
import random
import numpy as np
import datetime
from model.model_main import Model
from main.main_utils import read_config, write_yaml, override_configs, TeeLogger
from main.workers import Worker


rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
local_world_size = torch.cuda.device_count()


# read config
config_file = sys.argv[1]
config = read_config(config_file)
config = override_configs(dict(config), sys.argv[2:])


if config.get('disable_optimize_ddp', False):
    import torch._dynamo
    # temporary fix until singularity can support pytorch 2.3
    # https://github.com/pytorch/pytorch/issues/111636#issuecomment-1869471021
    torch._dynamo.config.optimize_ddp = False


# read relevant fields from config
out_dir = config['exec']['output_dir']
device = f'cuda:{local_rank}'
torch.cuda.set_device(device)
backend = config.get('dist_backend', 'nccl')
seed = config.get('seed', 42)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# cache config in rank 0
if rank == 0:
    try:
        repo = git.Repo(search_parent_directories=True)
        config['git_commit_id'] = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        pass
    os.makedirs(out_dir, exist_ok=True)
    write_yaml(config, os.path.join(out_dir, 'config.yaml'))


# setup logger and log basic info
logger = logging.getLogger(__name__)
logs_path = os.path.join(out_dir, 'logs', f'logs{rank}.txt')
if config.get('log_local_only', False):
    logs_path = os.path.join('logs', f'logs{rank}.txt')
logging.basicConfig(
    format=f"%(asctime)s - %(levelname)s - %(name)s - rank{rank} - %(message)s",
    stream=TeeLogger(logs_path),
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(f'{rank} {world_size} {local_rank} {local_world_size}')
logger.info(f'using backend {backend}')


def is_deepspeed(config):
    if 'train' not in config['exec']:
        return False
    return config['exec']['train']['forward_backward'].get('deepspeed', False)
if not is_deepspeed(config):
    master_uri = "tcp://%s:%s" % (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    torch.distributed.init_process_group(backend=backend,
                                        init_method=master_uri,
                                        world_size=world_size,
                                        rank=rank,
                                        timeout=datetime.timedelta(seconds=180000))


# create model, worker and run worker
model = Model(config['model'], rank, world_size, device)
worker = Worker(model,
                config['exec'],
                rank,
                world_size,
                device,
                local_rank=local_rank,
                local_world_size=local_world_size)
worker.run()
os._exit(0)
