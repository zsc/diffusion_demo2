"""Evaluate a manifest sequentially inside one reserved 16-GPU allocation.

Invoke with ordinary Python, not torchrun. Each evaluation owns a fresh process
 group: the GPU backend may hang when reinitializing NCCL in one worker process.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        raise RuntimeError('Run this coordinator with ordinary Python, not torchrun')
    root = Path(__file__).resolve().parent
    env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK']:
        env.pop(key, None)
    for entry in json.loads(Path(args.manifest).read_text()):
        output = Path(entry['output'])
        if (output / 'feature_scores.json').exists():
            continue
        subprocess.run(['bash', str(root / 'scripts/run_latent_gpu_group.sh'),
                        '--mode', 'eval', '--output', str(output), *entry['args']],
                       env=env, check=True)
        subprocess.run([sys.executable, str(root / 'scripts/score_pixel_features.py'),
                        str(output)], env=env, check=True)


if __name__ == '__main__':
    main()
