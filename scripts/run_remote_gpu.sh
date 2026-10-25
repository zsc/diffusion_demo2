#!/usr/bin/env bash
# Sync the project to a user-configured GPU host and copy the run back locally.
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/local_smoke.json}"
remote_host="${REMOTE_HOST:?Set REMOTE_HOST, for example gpu-host}"
remote_dir="${REMOTE_DIR:-/path/to/diffusion_demo2}"
run_id="$(date +%Y%m%d_%H%M%S)_$(basename "$config" .json)"
remote_python="${REMOTE_PYTHON:-python}"
rsync -az --exclude=.git --exclude=.venv --exclude=__pycache__ --exclude=runs --exclude=cache --exclude=checkpoints --exclude=results ./ "$remote_host:$remote_dir/"
printf -v remote_command 'cd %q && PYTHONPATH=src PYTHONUNBUFFERED=1 %q experiment.py --config %q --output %q' "$remote_dir" "$remote_python" "$config" "runs/$run_id"
mkdir -p runs/remote_logs
ssh "$remote_host" "$remote_command" 2>&1 | tee "runs/remote_logs/$run_id.log"
rsync -az "$remote_host:$remote_dir/runs/$run_id/" "runs/$run_id/"
printf 'Local results: %s/runs/%s\n' "$PWD" "$run_id"
