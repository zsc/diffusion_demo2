"""Regression checks for autoencoder scale, transfer and evaluation contracts."""
import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from diffusion_demo.core.data_loader import CelebASpritesDataset, tensor_to_pil
from diffusion_demo.core.models.vae import Discriminator, VAE, VQVAE, VectorQuantizer
from diffusion_demo.core.training.train_dit import load_vae, sample_images, train_dit_model
from diffusion_demo.core.training.train_vae import codebook_metrics, make_loaders, train_vae_model, vae_loss, vqvae_loss


class AutoencoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_vq_loss_scale_and_legacy_mode(self):
        for batch, size in ((1, 48), (3, 96)):
            x = torch.zeros(batch, 3, size, size)
            recon = torch.ones_like(x)
            loss, metrics = vqvae_loss(recon, x, torch.tensor(2.0), vq_beta=0.25)
            self.assertEqual(loss.item(), 1.5)
            self.assertEqual(metrics['recon_loss'], 1.0)
            old, _ = vqvae_loss(recon, x, torch.tensor(2.0), 0.25, 'sum')
            self.assertEqual(old.item(), 3 * size * size + 0.5)
        loss, _ = vae_loss(torch.ones_like(x), x, torch.zeros(3, 2, 4, 4), torch.zeros(3, 2, 4, 4))
        self.assertEqual(loss.item(), 3 * 96 * 96)
        with self.assertRaisesRegex(ValueError, 'shape'):
            vqvae_loss(x[:, :, :48, :48], x, torch.tensor(0.0))

    def test_chunked_quantizer_matches_dense_assignments_and_gradients(self):
        torch.manual_seed(9)
        module = VectorQuantizer(17, 5, distance_chunk_size=7)
        dense = copy.deepcopy(module)
        z = torch.randn(3, 5, 4, 4, requires_grad=True)
        reference_z = z.detach().clone().requires_grad_(True)
        flattened = reference_z.permute(0, 2, 3, 1).reshape(-1, 5)
        indices = (flattened.square().sum(1, keepdim=True) + dense.embeddings.weight.square().sum(1)
                   - 2 * flattened @ dense.embeddings.weight.T).argmin(1)
        quantized = dense.embeddings(indices).reshape(3, 4, 4, 5).permute(0, 3, 1, 2).contiguous()
        reference_loss = F.mse_loss(quantized, reference_z.detach()) + 0.25 * F.mse_loss(quantized.detach(), reference_z)
        reference_out = reference_z + (quantized - reference_z).detach()
        # FP32 search remains exact inside autocast; loss and STE gradients agree.
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out, loss, actual_indices = module(z)
        torch.testing.assert_close(actual_indices, indices)
        torch.testing.assert_close(out, reference_out)
        torch.testing.assert_close(loss, reference_loss)
        (out.square().mean() + loss).backward()
        (reference_out.square().mean() + reference_loss).backward()
        torch.testing.assert_close(z.grad, reference_z.grad)
        torch.testing.assert_close(module.embeddings.weight.grad, dense.embeddings.weight.grad)
        with torch.no_grad():
            module.embeddings.weight.zero_()
        self.assertTrue((module(z)[2] == 0).all())

    def test_known_unused_legacy_layers_are_ignored_but_other_keys_are_strict(self):
        model = VAE(3, [4, 8]).eval()
        old = dict(model.state_dict())
        for layer in ('down_mu', 'down_logvar'):
            old[f'encoder.{layer}.weight'] = torch.randn(3, 3, 3, 3)
            old[f'encoder.{layer}.bias'] = torch.randn(3)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'legacy.pth'
            torch.save(old, path)
            restored = load_vae(path, device='cpu')
            x = torch.randn(2, 3, 48, 48)
            torch.testing.assert_close(model.decode(model.encoder(x)[0]), restored.decode(restored.encoder(x)[0]))
        old['encoder.unknown.weight'] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
            model.load_state_dict(old, strict=True)
        self.assertFalse(any('down_mu' in key or 'down_logvar' in key for key in model.state_dict()))

    def test_spatial_transfer_keeps_all_learned_weights(self):
        for cls, options, small_grid, big_grid in (
            (VAE, dict(latent_dim=3, hidden_dims=[4, 8]), 6, 12),
            (VQVAE, dict(embedding_dim=3, num_embeddings=11, hidden_dims_enc=[4, 8], hidden_dims_dec=[8, 4]), 12, 24),
        ):
            source = cls(**options, image_size=48, latent_spatial_size=small_grid).eval()
            target = cls(**options, image_size=96, latent_spatial_size=big_grid).eval()
            target.load_state_dict(source.state_dict(), strict=True)
            self.assertEqual(target.get_latent_shape(2), (2, 3, big_grid, big_grid))
            with torch.no_grad():
                self.assertEqual(target(torch.randn(2, 3, 96, 96))[0].shape, (2, 3, 96, 96))
            for key in source.state_dict():
                torch.testing.assert_close(source.state_dict()[key], target.state_dict()[key])
            with self.assertRaisesRegex(ValueError, 'encoder grid'):
                target(torch.zeros(2, 3, 48, 48))

    def test_native_data_and_validation_first_are_explicit_and_rng_isolated(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'data.npy'
            data = np.zeros((10, 96, 96, 3), dtype=np.uint8)
            data[:, 0, 0, 0] = np.arange(10)
            np.save(path, data)
            train, val = make_loaders(2, 10, 7, 0.1, 0, False, str(path), 96, 'validation_first', 3)
            self.assertEqual(train.dataset.indices, list(range(3, 10)))
            self.assertEqual(val.dataset.indices, [0, 1, 2])
            state = torch.get_rng_state().clone()
            list(val)
            torch.testing.assert_close(state, torch.get_rng_state())
            with self.assertRaisesRegex(ValueError, 'uint8'):
                CelebASpritesDataset(cache_path=path, image_size=48)
            with self.assertRaisesRegex(ValueError, 'explicit data_path'):
                CelebASpritesDataset(image_size=96)
            with self.assertRaisesRegex(ValueError, 'explicit positive val_count'):
                make_loaders(2, 10, 7, 0.1, 0, False, str(path), 96, 'validation_first')

    def test_codebook_metrics_detect_collapsed_usage(self):
        self.assertEqual(codebook_metrics(torch.tensor([10, 0, 0, 0]), 'val'),
                         {'val_codes_used': 1, 'val_codebook_perplexity': 1.0})
        uniform = codebook_metrics(torch.ones(4), 'train')
        self.assertEqual(uniform['train_codes_used'], 4)
        self.assertAlmostEqual(uniform['train_codebook_perplexity'], 4.0)

    def test_gan_logits_are_compatible_and_extreme_loss_has_finite_gradients(self):
        model = Discriminator().eval()
        x = torch.randn(2, 3, 48, 48)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model.logits(x).sigmoid())
        logits = torch.tensor([-1000.0, 1000.0], requires_grad=True)
        loss = (F.softplus(-logits) + F.softplus(logits)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs() > 0).all())

    def test_preview_range_is_not_inferred_from_pixel_minimum(self):
        image = np.asarray(tensor_to_pil(torch.zeros(3, 2, 2)))
        self.assertTrue((image == 127).all())
        image = np.asarray(tensor_to_pil(torch.zeros(3, 2, 2), normalized=False))
        self.assertTrue((image == 0).all())

    def test_native96_vq_to_dit_checkpoint_and_sampling(self):
        data = np.random.default_rng(11).integers(0, 256, (6, 96, 96, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'data.npy'
            np.save(path, data)
            init = Path(root) / 'old48.pth'
            torch.save(VQVAE(3, 11, hidden_dims_enc=[4, 8], hidden_dims_dec=[8, 4]).state_dict(), init)
            options = dict(data_path=str(path), data_limit=6, split_mode='validation_first', val_count=2,
                           device='cpu', epochs=1, batch_size=2)
            result = train_vae_model(model_type='vqvae', embedding_dim=3, num_embeddings=11,
                                     vq_hidden_dims_enc=[4, 8], vq_hidden_dims_dec=[8, 4],
                                     image_size=96, latent_spatial_size=24, distance_chunk_size=31,
                                     init_checkpoint=str(init), output_dir=str(Path(root) / 'vq'), **options)
            metrics = json.loads(Path(result['metrics_path']).read_text())['epochs'][0]
            self.assertEqual(metrics['images_seen'], 4)
            self.assertEqual(metrics['optimizer_updates'], 2)
            self.assertEqual(metrics['skipped_updates'], 0)
            self.assertIn('val_codebook_perplexity', metrics)
            self.assertEqual(load_vae(result['best_model_path'], device='cpu').get_latent_shape(), (1, 3, 24, 24))
            dit = train_dit_model(result['best_model_path'], dit_hidden_size=16, dit_depth=1, dit_num_heads=2,
                                 dit_patch_size=2, ddpm_timesteps=4, sampling_steps=2, num_preview_samples=2,
                                 output_dir=str(Path(root) / 'dit'), **options)
            samples = sample_images(result['best_model_path'], dit['model_path'], num_samples=2,
                                    sampling_steps=2, device='cpu')
            self.assertEqual(samples.shape, (2, 3, 96, 96))
            self.assertTrue(torch.isfinite(samples).all())

    def test_two_rank_training_for_both_autoencoders(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            np.save(root / 'data.npy', np.zeros((10, 48, 48, 3), dtype=np.uint8))
            env = dict(os.environ, OMP_NUM_THREADS='2')
            if sys.platform == 'darwin':
                env['GLOO_SOCKET_IFNAME'] = 'lo0'
            for model_type in ('vae', 'vqvae'):
                config = dict(model_type=model_type, epochs=2, batch_size=4, data_limit=10,
                              validation_fraction=0.2, hidden_dims=[4, 8], latent_dim=3,
                              embedding_dim=3, num_embeddings=11, vq_hidden_dims_enc=[4, 8],
                              vq_hidden_dims_dec=[8, 4])
                config_path = root / f'{model_type}.json'
                config_path.write_text(json.dumps(config))
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                output = root / model_type
                run = subprocess.run([
                    sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=2',
                    '--master_addr=127.0.0.1', f'--master_port={port}', str(repo / 'vae_ddp_experiment.py'),
                    '--cpu', '--config', str(config_path), '--data', str(root / 'data.npy'), '--output', str(output)
                ], cwd=repo, env=env, capture_output=True, text=True, timeout=45)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                metrics = json.loads((output / 'metrics.json').read_text())['epochs']
                self.assertEqual(len(metrics), 2)
                self.assertEqual(metrics[-1]['world_size'], 2)
                self.assertEqual(metrics[-1]['images_seen'], 16)
                self.assertEqual(metrics[-1]['optimizer_updates'], 2)
                self.assertEqual(metrics[-1]['skipped_updates'], 0)
                self.assertTrue(torch.isfinite(load_vae(output / 'checkpoints/last.pth', device='cpu')
                                              .decode(torch.zeros(1, 3, 4, 4))).all())

    def test_cli_uses_heldout_core_trainer_and_metadata(self):
        from cmdline_train import train_vae_cmdline
        data = np.random.default_rng(0).integers(0, 256, (6, 48, 48, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root, patch('diffusion_demo.core.data_loader.load_cached_images', return_value=data):
            result = train_vae_cmdline('vqvae', epochs=1, batch_size=2, data_limit=6, device='cpu',
                                       embedding_dim=3, num_embeddings=11, vq_hidden_dims_enc=[4, 8],
                                       vq_hidden_dims_dec=[8, 4], validation_fraction=0.34, output_dir=root,
                                       use_gan=True)
            payload = torch.load(result['model_path'], weights_only=True)
            self.assertEqual(payload['model_type'], 'vqvae')
            self.assertEqual(payload['model_config']['image_size'], 48)
            self.assertEqual(result['metrics']['validation_images'], 2)
            self.assertEqual(result['metrics']['epochs'][0]['generated_images'], [])
            self.assertIn('val_recon_mse', result['metrics']['epochs'][0])
            self.assertIn('g_adv_loss', result['metrics']['epochs'][0])
            html = Path(result['html_path']).read_text()
            self.assertNotIn('Generated Images (Random Latent Sampling)', html)
            self.assertIn('Validation PSNR', html)
            with self.assertRaisesRegex(ValueError, 'already contains a run'):
                train_vae_model(epochs=1, data_limit=6, device='cpu', output_dir=root)


if __name__ == '__main__':
    unittest.main()
