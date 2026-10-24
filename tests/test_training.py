import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from diffusion_demo.core.data_loader import parse_sprite_sheet

from diffusion_demo.core.models.dit import MultiHeadAttention, create_dit_for_latent
from diffusion_demo.core.models.vae import VAE, VQVAE
from diffusion_demo.core.training.train_dit import DDPM, load_vae, sample_images, sample_dit, train_dit_model, latent_batch
from diffusion_demo.core.training.train_vae import make_loaders, train_vae_model
from diffusion_demo.utils.torch_env import MixedPrecision, setup_seed


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sprite_parser_uses_all_columns_in_row_order(self):
        sheet = np.zeros((96, 144, 3), dtype=np.uint8)
        for row in range(2):
            for col in range(3):
                sheet[row*48:(row+1)*48, col*48:(col+1)*48] = row*3+col
        images = parse_sprite_sheet(Image.fromarray(sheet))
        self.assertEqual(images.shape, (6, 48, 48, 3))
        np.testing.assert_array_equal(images[:, 0, 0, 0], np.arange(6))
        with self.assertRaises(ValueError):
            parse_sprite_sheet(Image.new('RGB', (49, 48)))

    def test_sdpa_matches_attention_and_gradients(self):
        module = MultiHeadAttention(32, 4)
        x = torch.randn(2, 16, 32, requires_grad=True)
        q, k, v = module.qkv(x).reshape(2, 16, 3, 4, 8).permute(2, 0, 3, 1, 4)
        expected = module.proj(((q @ k.transpose(-2, -1) / 8**0.5).softmax(-1) @ v)
                               .transpose(1, 2).reshape(2, 16, 32))
        actual = module(x)
        torch.testing.assert_close(actual, expected)
        reference_grad = torch.autograd.grad(expected.sum(), x, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual.sum(), x)[0]
        torch.testing.assert_close(actual_grad, reference_grad)

    def test_ddim_oracle_recovers_clean_signal(self):
        for schedule in ('linear', 'cosine'):
            ddpm = DDPM(10, schedule, 'cpu')
            class Oracle(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.anchor = torch.nn.Parameter(torch.tensor(0.))
                def forward(self, x, t):
                    alpha = ddpm.alphas_cumprod[t].view(-1, 1, 1, 1)
                    return (x - alpha.sqrt() * 0.25) / (1 - alpha).sqrt()
            for steps in (1, 4, 10):
                result = ddpm.ddim_sample_loop(Oracle(), (2, 3, 4, 4), steps)
                torch.testing.assert_close(result, torch.full_like(result, 0.25), atol=5e-4, rtol=5e-4)

    def test_legacy_autoencoder_architecture_inferred(self):
        with tempfile.TemporaryDirectory() as root:
            for model in (VAE(7, [8, 16]), VQVAE(5, 13, hidden_dims_enc=[8, 16], hidden_dims_dec=[16, 8])):
                path = Path(root) / 'old.pth'
                torch.save(model.state_dict(), path)
                loaded = load_vae(path, device='cpu')
                for key, weight in model.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], weight)
                self.assertEqual(loaded.decode(loaded.encode(torch.zeros(2, 3, 48, 48))).shape, (2, 3, 48, 48))

    def test_seeded_split_is_disjoint_and_repeatable(self):
        data = np.zeros((20, 48, 48, 3), dtype=np.uint8)
        with patch('diffusion_demo.core.data_loader.load_cached_images', return_value=data):
            first = make_loaders(4, 20, 42, 0.2, 0, False)
            second = make_loaders(4, 20, 42, 0.2, 0, False)
        self.assertEqual(first[0].dataset.indices, second[0].dataset.indices)
        self.assertFalse(set(first[0].dataset.indices) & set(first[1].dataset.indices))

    def test_end_to_end_checkpoint_sampling(self):
        data = np.random.default_rng(0).integers(0, 256, (8, 48, 48, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root, patch('diffusion_demo.core.data_loader.load_cached_images', return_value=data):
            vae = train_vae_model(epochs=1, batch_size=2, data_limit=8, latent_dim=4,
                                  validation_fraction=0.25, device='cpu', output_dir=str(Path(root) / 'vae'))
            dit = train_dit_model(vae['model_path'], epochs=1, batch_size=2, data_limit=8,
                                  dit_hidden_size=32, dit_depth=1, dit_num_heads=4,
                                  ddpm_timesteps=10, ddpm_schedule='cosine', sampling_steps=4,
                                  validation_fraction=0.25, num_preview_samples=2,
                                  device='cpu', output_dir=str(Path(root) / 'dit'))
            # Intentionally wrong UI defaults must be overridden by checkpoint metadata.
            samples = sample_images(vae['model_path'], dit['model_path'], num_samples=2,
                                    dit_hidden_size=99, ddpm_timesteps=1000,
                                    sampling_steps=4, device='cpu')
            repeat = sample_images(vae['model_path'], dit['model_path'], num_samples=2,
                                   sampling_steps=4, device='cpu')
            self.assertEqual(samples.shape, (2, 3, 48, 48))
            self.assertTrue(torch.isfinite(samples).all())
            torch.testing.assert_close(samples, repeat)
            payload = torch.load(dit['model_path'], weights_only=True)
            restored = create_dit_for_latent(**payload['model_config']).eval()
            restored.load_state_dict(payload['ema_state_dict'])
            restored_vae = load_vae(vae['model_path'], device='cpu')
            setup_seed(0)
            expected = sample_dit(restored, restored_vae,
                                  DDPM(10, 'cosine', 'cpu'), num_samples=2, latent_channels=4,
                                  sampling_steps=4, latent_mean=payload['latent_mean'],
                                  latent_std=payload['latent_std'])
            torch.testing.assert_close(samples, expected)
            metrics = json.loads(Path(dit['metrics_path']).read_text())
            self.assertIn('val_ema_noise_mse', metrics['epochs'][0])
            self.assertGreater(metrics['epochs'][0]['optimizer_updates'], 0)
            self.assertTrue(Path(vae['best_model_path']).exists())

    def test_cached_posterior_is_resampled_and_normalization_inverts(self):
        vae = VAE(2, [4, 8]).eval()
        mean = torch.randn(3, 2, 4, 4)
        logvar = torch.full_like(mean, -0.5)
        center = torch.tensor([0.2, -0.3]).reshape(1, 2, 1, 1)
        scale = torch.tensor([2.0, 0.4]).reshape(1, 2, 1, 1)
        def draw(seed):
            return latent_batch((mean, logvar), vae, 'cpu', MixedPrecision('cpu'), True,
                                center, scale, torch.Generator().manual_seed(seed))
        first, second = draw(1), draw(2)
        self.assertFalse(torch.equal(first, second))
        expected = mean + (logvar / 2).exp() * torch.randn(mean.shape, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(first * scale + center, expected)

    def test_precision_falls_back_on_cpu(self):
        amp = MixedPrecision('cpu', 'auto')
        self.assertEqual(amp.precision, 'fp32')
        with self.assertRaises(ValueError):
            MixedPrecision('cpu', 'invalid')


if __name__ == '__main__':
    unittest.main()
