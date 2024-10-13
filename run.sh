#!/bin/bash

# Command:
# > source run.sh path_to_config.yaml

export PYTHONPATH="${PYTHONPATH}:$(pwd)/corgee"

if [ -z "$DIST_NUM_NODES" ]; then
    torchrun --standalone --nnodes=1 --nproc-per-node=$(python -c "import torch; print(torch.cuda.device_count())") corgee/main/run.py $@
else
    torchrun --nnodes=$DIST_NUM_NODES --nproc-per-node=$(python -c "import torch; print(torch.cuda.device_count())") corgee/main/run.py $@
fi
