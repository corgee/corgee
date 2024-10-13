#!/bin/bash

# Command:
# > source run.sh path_to_config.yaml

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code:$(pwd)/code/third_party/mteb_beir"
export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(pwd)/code/data/cython/build
export NCCL_DEBUG=WARN

# if [ -z "$DIST_NUM_NODES" ]; then
#     torchrun --standalone --nnodes=1 --nproc-per-node=$(python -c "import torch; print(torch.cuda.device_count())") code/main/run.py $@
# else
#     torchrun --standalone --nnodes=$DIST_NUM_NODES --nproc-per-node=$(python -c "import torch; print(torch.cuda.device_count())") code/main/run.py $@
# fi
torchrun code/main/run.py $@  ## For running a single process
