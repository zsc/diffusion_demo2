"""Distributed VAE/VQ-VAE training; config batch_size is the global batch."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import torch
import torch.distributed as dist
from diffusion_demo.core.training.train_vae import train_vae_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--cpu', action='store_true', help='Gloo correctness check only')
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    local_rank = int(os.environ['LOCAL_RANK'])
    if args.cpu:
        device, backend = 'cpu', 'gloo'
    else:
        torch.cuda.set_device(local_rank)
        device, backend = f'cuda:{local_rank}', 'nccl'
    dist.init_process_group(backend)
    try:
        config = json.loads(Path(args.config).read_text())
        result = train_vae_model(**(config | dict(output_dir=args.output, data_path=args.data,
                                                   device=device, distributed=True)))
        if dist.get_rank() == 0:
            (Path(args.output) / 'result.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
