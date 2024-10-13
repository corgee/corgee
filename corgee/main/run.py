import datetime
import logging
import os
import random
import sys

import numpy as np
import torch
from main.main_utils import TeeLogger, override_configs, read_config
from main.workers import Worker
from model.model_main import Model

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
local_world_size = torch.cuda.device_count()


# read config
config_file = sys.argv[1]
config = read_config(config_file)
config = override_configs(dict(config), sys.argv[2:])


# read relevant fields from config
out_dir = config["exec"]["output_dir"]
device = f"cuda:{local_rank}"
torch.cuda.set_device(device)
backend = config.get("dist_backend", "nccl")
seed = config.get("seed", 42)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# setup logger and log basic info
logger = logging.getLogger(__name__)
logs_path = os.path.join(out_dir, "logs", f"logs{rank}.txt")
if config.get("log_local_only", False):
    logs_path = os.path.join("logs", f"logs{rank}.txt")
logging.basicConfig(
    format=f"%(asctime)s - %(levelname)s - %(name)s - rank{rank} - %(message)s",
    stream=TeeLogger(logs_path),
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(f"{rank} {world_size} {local_rank} {local_world_size}")
logger.info(f"using backend {backend}")


master_uri = "tcp://%s:%s" % (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
torch.distributed.init_process_group(
    backend=backend,
    init_method=master_uri,
    world_size=world_size,
    rank=rank,
    timeout=datetime.timedelta(seconds=180000),
)

# create model, worker and run worker
model = Model(config["model"], rank, world_size, device)
worker = Worker(
    model,
    config["exec"],
    rank,
    world_size,
    device,
    local_rank=local_rank,
    local_world_size=local_world_size,
)
worker.run()
os._exit(0)
