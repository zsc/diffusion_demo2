"""Measure actual CUDA training and sampling costs on one VAE/DiT checkpoint."""
import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from diffusion_demo.core.data_loader import get_dataloader, tensor_to_pil
from diffusion_demo.core.models.dit import MultiHeadAttention, create_dit_for_latent
from diffusion_demo.core.training.train_dit import DDPM, load_vae, sample_dit
from diffusion_demo.utils.torch_env import MixedPrecision, setup_seed, get_torch_info


def manual_attention(self, x):
    b, n, c = x.shape
    q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    weights = (q @ k.transpose(-2, -1) * self.head_dim**-0.5).softmax(-1)
    return self.proj((weights @ v).transpose(1, 2).reshape(b, n, c))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vae', required=True)
    parser.add_argument('--dit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA required for this benchmark')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    setup_seed(42)
    vae = load_vae(args.vae, device='cuda')
    payload = torch.load(args.dit, map_location='cpu', weights_only=True)
    original = create_dit_for_latent(**payload['model_config']).cuda()
    original.load_state_dict(payload['model_state_dict'])
    ddpm = DDPM(device='cuda', **payload['diffusion_config'])
    images = next(iter(get_dataloader(args.batch_size, data_limit=128, shuffle=False))).cuda()
    center, scale = payload['latent_mean'].cuda(), payload['latent_std'].cuda()
    with torch.no_grad():
        mean, logvar = vae.encoder(images)
    trials = []
    # Repeat all modes to expose interference from other GPU users.
    modes = [(False, 'fp32', False), (True, 'fp32', False), (True, 'auto', False), (True, 'auto', True)]
    for repeat in range(3):
        for cached, precision, manual in (modes if repeat % 2 == 0 else list(reversed(modes))):
            setup_seed(42)
            model = copy.deepcopy(original).train()
            if manual:
                for module in model.modules():
                    if isinstance(module, MultiHeadAttention):
                        module.forward = types.MethodType(manual_attention, module)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            amp = MixedPrecision('cuda', precision)
            def step():
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), amp.autocast():
                    mu, lv = (mean, logvar) if cached else vae.encoder(images)
                    z = (mu.float() + (lv.float() * 0.5).exp() * torch.randn_like(mean) - center) / scale
                t = torch.randint(0, ddpm.timesteps, (len(z),), device='cuda')
                with amp.autocast():
                    loss = ddpm.p_losses(model, z, t)
                amp.step(loss, optimizer, model.parameters())
            for _ in range(10):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.steps):
                step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            record = dict(repeat=repeat, cached=cached, precision=amp.precision,
                          attention='manual' if manual else 'sdpa', seconds=seconds,
                          images_per_second=len(images) * args.steps / seconds,
                          peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20)
            trials.append(record)
            print(json.dumps(record), flush=True)
            del model, optimizer, amp
    original.load_state_dict(payload['ema_state_dict'])
    original.eval()
    sampling = []
    amp = MixedPrecision('cuda', 'auto')
    for steps in (None, 50):
        setup_seed(42)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with amp.autocast():
            samples = sample_dit(original, vae, ddpm, num_samples=16,
                                 latent_channels=payload['model_config']['latent_channels'],
                                 sampling_steps=steps, latent_mean=center, latent_std=scale)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        tensor_to_pil(samples.float()).save(root / f'samples_{steps or ddpm.timesteps}.png')
        sampling.append(dict(method='ddpm' if steps is None else 'ddim', steps=steps or ddpm.timesteps,
                             seconds=seconds))
    (root / 'benchmark.json').write_text(json.dumps(dict(environment=get_torch_info(), trials=trials,
                                                        sampling=sampling, args=vars(args)), indent=2))


if __name__ == '__main__':
    main()
