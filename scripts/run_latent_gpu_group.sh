#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
train_python=${TRAIN_PYTHON:-/usr/local/bin/python3}
"$train_python" -c 'import torch; assert torch.cuda.device_count() == 16'
env -u WORLD_SIZE -u RANK -u LOCAL_RANK "$train_python" -m torch.distributed.run \
 --standalone --nnodes=1 --nproc_per_node=16 latent_ddp_experiment.py "$@"
