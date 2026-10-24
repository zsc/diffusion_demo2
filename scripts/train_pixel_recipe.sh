#!/usr/bin/env bash
# Reserve one 16-PPU node; train the guide and main predictor sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."
data_file=$1
eval_assets=$2
output_dir=$3
train_python=${TRAIN_PYTHON:-/usr/local/bin/python3}
QUALITY_EVAL=0 bash scripts/run_ppu_group.sh configs/pixel_ddp_guide_b1024_lr2e4.json "$output_dir/guide" "$data_file" "$eval_assets"
QUALITY_EVAL=0 bash scripts/run_ppu_group.sh configs/pixel_ddp_b1024_lr4e4.json "$output_dir/main" "$data_file" "$eval_assets"
export PYTHONPATH="$PWD/src:$eval_assets/packages" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
env -u WORLD_SIZE -u RANK -u LOCAL_RANK "$train_python" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=16 --master_addr=127.0.0.1 --master_port=29574 \
  evaluate_pixel.py --checkpoint "$output_dir/main/ema_008192000.pt" \
  --guide-checkpoint "$output_dir/guide/ema_002048000.pt" --guidance 2 --guidance-start .1 --guidance-end .8 \
  --seed 7777 --data "$data_file" --weights-dir "$eval_assets" --output "$output_dir/quality"
"$train_python" sample_pixel.py --checkpoint "$output_dir/main/ema_008192000.pt" \
  --guide-checkpoint "$output_dir/guide/ema_002048000.pt" --guidance 2 --guidance-start .1 --guidance-end .8 \
  --seed 2027 --count 64 --precision bf16 --output "$output_dir/samples"
