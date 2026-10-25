#!/usr/bin/env bash
# A CLUSTER task must reserve one full 16-GPU node for this launcher.
set -euo pipefail
cd "$(dirname "$0")/.."
train_config=$1
output_dir=$2
data_file=$3
eval_assets=$4
resume_args=()
if [[ $# -ge 5 ]]; then
  resume_args=(--resume "$5")
fi
if [[ "${ALLOW_RESUME_LR_CHANGE:-0}" == 1 ]]; then
  [[ $# -ge 5 ]] || { echo 'LR change requires a resume checkpoint' >&2; exit 1; }
  resume_args+=(--allow-resume-lr-change)
fi
if [[ "${ALLOW_RESUME_TIME_CHANGE:-0}" == 1 ]]; then
  [[ $# -ge 5 ]] || { echo 'Time distribution change requires a resume checkpoint' >&2; exit 1; }
  resume_args+=(--allow-resume-time-change)
fi
if [[ "${ALLOW_RESUME_BATCH_CHANGE:-0}" == 1 ]]; then
  [[ $# -ge 5 ]] || { echo 'Batch change requires a resume checkpoint' >&2; exit 1; }
  resume_args+=(--allow-resume-batch-change)
fi
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  [[ ${#resume_args[@]} == 0 ]] || { echo 'Choose resume or initialization, not both' >&2; exit 1; }
  resume_args=(--init-from "$INIT_CHECKPOINT")
fi
train_python=${TRAIN_PYTHON:-/usr/local/bin/python3}
export PYTHONPATH="$PWD/src:$eval_assets/packages"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
"$train_python" -c 'import torch; print("visible GPUs",torch.cuda.device_count()); assert torch.cuda.device_count()==16'
mkdir -p "$output_dir"
env -u WORLD_SIZE -u RANK -u LOCAL_RANK "$train_python" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=16 --master_addr=127.0.0.1 --master_port=29573 \
  pixel_ddp_experiment.py --config "$train_config" --output "$output_dir" --data "$data_file" "${resume_args[@]}" \
  > "$output_dir/train.log" 2>&1
if [[ "${QUALITY_EVAL:-1}" == 1 ]]; then
  env -u WORLD_SIZE -u RANK -u LOCAL_RANK "$train_python" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=16 --master_addr=127.0.0.1 --master_port=29573 \
    evaluate_pixel.py --checkpoint "$output_dir/last.pt" --data "$data_file" \
    --weights-dir "$eval_assets" --output "$output_dir/quality" \
    > "$output_dir/quality.log" 2>&1
fi
printf 'GROUP_FINISHED\n'
