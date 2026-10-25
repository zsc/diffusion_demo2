"""Matched-exposure full CelebA study initialized from one fixed best EMA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from prepare_celeba96_full import array_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--tag', required=True,
                        choices=['control30k', 'full', 'full_batch4096', 'full_lr2e5'])
    args = parser.parse_args()
    root, tag = Path(args.root), args.tag
    output = root / 'round96_full_20261025' / tag
    parent = root / 'round96_cycle2_20261025/stratified/ema_032768000.pt'
    config_path = Path('configs') / f'pixel96_full_{tag}.json'
    config = json.loads(config_path.read_text())
    assert not output.exists(), 'Never overwrite an experiment'
    base_path = root / 'data96/celeba96.npy'
    data = base_path if tag == 'control30k' else root / 'data96_full/celeba96.npy'
    deadline = time.monotonic() + 1200
    while not data.with_suffix('.json').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Full training cache is not ready')
        time.sleep(10)
    meta = json.loads(data.with_suffix('.json').read_text())
    base_meta = json.loads(base_path.with_suffix('.json').read_text())
    array, base = np.load(data, mmap_mode='r'), np.load(base_path, mmap_mode='r')
    count = 30000 if tag == 'control30k' else 162770
    assert array.shape == (count + 1000, 96, 96, 3)
    assert meta['images']['validation'] == base_meta['images']['validation']
    assert len(meta['images']['train']) == count
    assert not set(meta['images']['train']).intersection(meta['images']['validation'])
    assert np.array_equal(array[:1000], base[:1000]), 'Validation pixels changed'
    assert array_hash(array) == meta['array_sha256'], 'Training cache checksum mismatch'
    if tag != 'control30k':
        assert meta['base_array_sha256'] == base_meta['array_sha256']
        assert meta['images']['train'][:30000] == base_meta['images']['train']
        assert meta['validation_byte_identical'] and meta['base_training_byte_identical']
    recipe = dict(training_images=count, validation_images=1000,
                  validation_sha256=array_hash(array[:1000]), data_sha256=meta['array_sha256'],
                  initial_checkpoint=str(parent), exposure_budget=config['max_images'],
                  optimizer='fresh in all arms', ema='same source EMA, fresh age in all arms',
                  best_of_n=False)
    del array, base
    output.mkdir(parents=True)
    (output / 'recipe.json').write_text(json.dumps(recipe, indent=2))
    print(json.dumps(recipe), flush=True)
    env = dict(os.environ, QUALITY_EVAL='0', INIT_CHECKPOINT=str(parent),
               ALLOW_RESUME_LR_CHANGE='0', ALLOW_RESUME_TIME_CHANGE='0', ALLOW_RESUME_BATCH_CHANGE='0')
    subprocess.run(['bash', 'scripts/run_gpu_group.sh', str(config_path), str(output),
                    str(data), str(root / 'eval_assets')], env=env, check=True, timeout=7200)
    assert json.loads((output / 'completed.json').read_text())['images_seen'] == 16384000
    assert json.loads((output / 'split.json').read_text())['data_sha256'] == meta['array_sha256']
    rows = []
    for images in [4096000, 8192000, 16384000]:
        for seed in [3333, 7777]:
            rows.append(dict(checkpoint=f'round96_full_20261025/{tag}/ema_{images:09d}.pt',
                             label=f'{tag}_{images}_g2_seed{seed}', guidance=2, seed=seed))
    rows.append(dict(checkpoint=f'round96_full_20261025/{tag}/ema_016384000.pt',
                     label=f'{tag}_16384000_g1_seed3333', guidance=1, seed=3333))
    manifest = output / f'{tag}_evaluation.json'
    manifest.write_text(json.dumps(rows, indent=2))
    # Evaluator intentionally reads the original cache: identical real validation set.
    subprocess.run([sys.executable, '-u', 'scripts/evaluate_pixel_continuation.py',
                    '--root', str(root), '--manifest', str(manifest),
                    '--output', str(root / 'round96_full_eval_20261025')], check=True, timeout=3600)
    (output / 'pipeline_completed.json').write_text(json.dumps(dict(
        additional_exposures=16384000, unfiltered_evaluations=len(rows), best_of_n=False)))


if __name__ == '__main__':
    main()
