#!/usr/bin/env bash
# Run from a tmux-created shell if direct SSH is blocked by the desktop network context.
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/desk_smoke.json}"
remote_dir=/home/zsc/Downloads/diffusion_demo2_codex
run_id="$(date +%Y%m%d_%H%M%S)_$(basename "$config" .json)"
remote_python=/home/zsc/miniconda3/envs/py312/bin/python
rsync -az --exclude=.git --exclude=.venv --exclude=__pycache__ --exclude=runs --exclude=cache --exclude=checkpoints --exclude=results ./ "desk:$remote_dir/"
printf -v remote_command 'cd %q && PYTHONPATH=src PYTHONUNBUFFERED=1 %q experiment.py --config %q --output %q' "$remote_dir" "$remote_python" "$config" "runs/$run_id"
mkdir -p runs/desk_logs
ssh desk "$remote_command" 2>&1 | tee "runs/desk_logs/$run_id.log"
rsync -az "desk:$remote_dir/runs/$run_id/" "runs/$run_id/"
printf 'Local results: %s/runs/%s\n' "$PWD" "$run_id"
