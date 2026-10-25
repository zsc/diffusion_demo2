#!/usr/bin/env bash
# Reserve one full 16-GPU CLUSTER group before invoking this launcher.
set -euo pipefail
cd "$(dirname "$0")/.."
train_python=${TRAIN_PYTHON:-/usr/local/bin/python3}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
"$train_python" -c 'import torch; assert torch.cuda.device_count() == 16, "Reserve exactly 16 GPUs"'
mkdir -p "$2"
env -u WORLD_SIZE -u RANK -u LOCAL_RANK "$train_python" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=16 --master_addr=127.0.0.1 --master_port=29574 \
  vae_ddp_experiment.py --config "$1" --output "$2" --data "$3" > "$2/train.log" 2>&1
