"""Small 2x2 VQ-VAE diagnostic: loss scale versus spatial bottleneck (not a full quality benchmark)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from diffusion_demo.core.training.train_vae import train_vae_model
from diffusion_demo.utils.torch_env import get_torch_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--data-limit', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source_root = Path(__file__).resolve().parents[1]
    environment = get_torch_info()
    environment['source_sha256'] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((source_root / 'src').rglob('*.py'))}
    (root / 'environment.json').write_text(json.dumps(environment, indent=2))
    summary = dict(purpose='Short diagnostic only; one seed, no generative FID or hardware speed comparison', runs={})
    for grid in (4, 12):
        for reduction in ('sum', 'mean'):
            name = f'grid{grid}_{reduction}'
            result = train_vae_model(
                model_type='vqvae', device=args.device, epochs=args.epochs, batch_size=32,
                data_limit=args.data_limit, validation_fraction=0.125, seed=args.seed, lr=2e-4,
                embedding_dim=16, num_embeddings=128, vq_hidden_dims_enc=[32, 64],
                vq_hidden_dims_dec=[64, 32], latent_spatial_size=grid, vq_recon_reduction=reduction,
                output_dir=str(root / name),
            )
            history = json.loads(Path(result['metrics_path']).read_text())['epochs']
            best = min(history, key=lambda e: e['val_recon_mse'])
            summary['runs'][name] = dict(best=best, final=history[-1], artifacts=result)
            (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v['best'] for k, v in summary['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
