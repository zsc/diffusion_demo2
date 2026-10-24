import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from diffusion_demo.core.models.pixel_jit import PixelJiT, flow_loss, heun_sample


class PixelJiTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_clean_prediction_velocity_loss_and_gradient(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.tensor(0.2))

            def forward(self, x, t):
                return self.p.expand_as(x)

        model = Constant()
        x, noise = torch.randn(3, 3, 8, 8), torch.randn(3, 3, 8, 8)
        t = torch.tensor([0.0, 0.3, 0.99])
        loss = flow_loss(model, x, t, noise)
        t4 = t[:, None, None, None]
        z = t4 * x + (1 - t4) * noise
        vpred = (model(z, t) - z) / (1 - t4).clamp_min(.05)
        target = (x - z) / (1 - t4).clamp_min(.05)
        reference = (vpred - target).square().mean()
        torch.testing.assert_close(loss, reference)
        g = torch.autograd.grad(loss, model.p)[0]
        rg = torch.autograd.grad(reference, model.p)[0]
        torch.testing.assert_close(g, rg)

    def test_heun_oracle_noise_to_data_and_final_euler(self):
        clean = torch.randn(2, 3, 8, 8)
        class Oracle(torch.nn.Module):
            def forward(self, x, t):
                return clean
        # With dt >= t_eps, the exact straight path reaches data at t=1.
        actual = heun_sample(Oracle(), torch.randn_like(clean), steps=10)
        torch.testing.assert_close(actual, clean, atol=1e-5, rtol=1e-5)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = np.random.default_rng(7).integers(0, 256, (10, 48, 48, 3), dtype=np.uint8)
            np.save(root / "data.npy", data)
            config = dict(model=dict(width=32, depth=1, heads=4, bottleneck=8,
                                     conv_layers=0, conv_channels=8),
                          device="cpu", precision="fp32", batch_size=2, steps=4, lr=1e-4,
                          val_count=2, sample_steps=2, sample_every=2, checkpoint_every=2, log_every=2)
            conf = root / "config.json"
            def run(name, steps, resume=None):
                config["steps"] = steps
                conf.write_text(json.dumps(config))
                cmd = [sys.executable, "pixel_experiment.py", "--config", str(conf), "--output", str(root/name),
                       "--data", str(root/"data.npy")]
                if resume:
                    cmd.extend(["--resume", str(resume)])
                result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            run("full", 4)
            run("split", 2)
            run("split", 4, root/"split"/"last.pt")
            a = torch.load(root/"full"/"last.pt", weights_only=False)
            b = torch.load(root/"split"/"last.pt", weights_only=False)
            for group in ["model", "ema"]:
                for key in a[group]:
                    torch.testing.assert_close(a[group][key], b[group][key], rtol=0, atol=0)
            self.assertTrue((root/"split"/"completed.json").exists())


if __name__ == "__main__":
    unittest.main()
