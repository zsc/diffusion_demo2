import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from diffusion_demo.core.models.pixel_jit import PixelJiT, flow_loss
from pixel_data import split_indices
from pixel_transfer import inherit_pixel_weights


class PixelTransferTests(unittest.TestCase):
    def test_resolution_transfer_copies_every_parameter_and_regenerates_positions(self):
        torch.set_num_threads(1)
        cfg = dict(image_size=48, width=32, depth=1, heads=4, bottleneck=8, conv_channels=8, conv_layers=1)
        source = PixelJiT(**cfg)
        # Nonzero output/gates exercise the inherited computation, not zero initialization.
        with torch.no_grad():
            for p in source.parameters():
                p.uniform_(-.1, .1)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "source.pt"
            torch.save(dict(ema=source.state_dict(), config={"model": cfg}), checkpoint)
            for scale in [1, .5]:
                target = PixelJiT(**{**cfg, "image_size": 96, "position_scale": scale})
                report = inherit_pixel_weights(target, checkpoint)
                self.assertEqual(report["learned_parameters_reinitialized"], 0)
                for k, p in source.state_dict().items():
                    torch.testing.assert_close(target.state_dict()[k], p, rtol=0, atol=0)
                self.assertEqual(target.pos.shape[1], 24 * 24)
                if scale == .5:
                    torch.testing.assert_close(target.pos.reshape(1, 24, 24, 32)[:, ::2, ::2],
                                               source.pos.reshape(1, 12, 12, 32))
                clean = torch.randn(1, 3, 96, 96)
                loss = flow_loss(target, clean, torch.tensor([.4]), torch.randn_like(clean))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(target.out.weight.grad.abs().sum().item(), 0)
            with self.assertRaisesRegex(ValueError, "patch_size"):
                inherit_pixel_weights(PixelJiT(**{**cfg, "image_size": 96, "patch_size": 8}), checkpoint)

    def test_96_training_and_sampling_use_target_shape_and_fresh_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arch = dict(image_size=48, width=32, depth=1, heads=4, bottleneck=8, conv_channels=8, conv_layers=0)
            torch.save(dict(ema=PixelJiT(**arch).state_dict(), config={"model": arch}, images_seen=8192000), root / "source.pt")
            cfg = dict(model={**arch, "image_size": 96, "position_scale": .5}, device="cpu", precision="fp32",
                       global_batch=2, lr=1e-4, warmup_images=2, max_images=2, eval_every_images=2,
                       val_count=2, split_mode="validation_first", sample_steps=1, log_every=1, cpu_threads=1)
            (root / "config.json").write_text(json.dumps(cfg))
            np.save(root / "data.npy", np.random.default_rng(1).integers(0, 256, (6, 96, 96, 3), dtype=np.uint8))
            env = {**os.environ, "PYTHONPATH": "src:."}
            cmd = [sys.executable, "pixel_ddp_experiment.py", "--config", str(root / "config.json"),
                   "--data", str(root / "data.npy"), "--output", str(root / "run"), "--init-from", str(root / "source.pt")]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            state = torch.load(root / "run/last.pt", weights_only=False)
            self.assertEqual(state["step"], 1)
            self.assertEqual(state["images_seen"], 2)
            self.assertEqual(state["initialization"]["source_images_seen"], 8192000)
            p = subprocess.run([sys.executable, "sample_pixel.py", "--checkpoint", str(root / "run/last.pt"),
                                "--device", "cpu", "--count", "2", "--steps", "1", "--output", str(root / "samples")],
                               env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual(np.load(root / "samples/samples.npy").shape, (2, 3, 96, 96))
            train, val = split_indices(6, cfg)
            self.assertEqual(train.tolist(), [2, 3, 4, 5])
            self.assertEqual(val.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
