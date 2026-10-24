"""Exercise distributed RNG, synchronization, checkpointing, and exact resume."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from pixel_ddp_experiment import validate_resume_config


class PixelDDPTests(unittest.TestCase):
    def test_lr_change_is_explicit_and_does_not_relax_other_checks(self):
        previous = {"lr": 2e-4, "global_batch": 1024}
        with self.assertRaisesRegex(ValueError, "changed lr"):
            validate_resume_config(previous, {**previous, "lr": 1e-4})
        validate_resume_config(previous, {**previous, "lr": 1e-4}, True)
        for lr in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                validate_resume_config(previous, {**previous, "lr": lr}, True)
        with self.assertRaisesRegex(ValueError, "global_batch"):
            validate_resume_config(previous, {"lr": 1e-4, "global_batch": 512}, True)

    def test_two_rank_resume_matches_continuous_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = dict(model=dict(width=32, depth=1, heads=4, bottleneck=8, conv_layers=0, conv_channels=8),
                       device="cpu", precision="fp32", global_batch=4, lr=1e-4, warmup_images=8,
                       ema_reference_batch=4, ema_reference_decay=.99, val_count=2, max_images=16,
                       eval_every_images=8, sample_steps=2, log_every=2, seed=42, cpu_threads=1)
            np.save(root / "data.npy", np.random.default_rng(7).integers(0, 256, (10, 48, 48, 3), dtype=np.uint8))
            def run(name, images, resume=False, allow_lr_change=False):
                cfg["max_images"] = images
                (root / "config.json").write_text(json.dumps(cfg))
                with socket.socket() as s:
                    s.bind(("127.0.0.1", 0))
                    port = s.getsockname()[1]
                cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
                       "--master_addr=127.0.0.1", f"--master_port={port}", "pixel_ddp_experiment.py",
                       "--config", str(root / "config.json"), "--data", str(root / "data.npy"),
                       "--output", str(root / name)]
                if resume:
                    cmd.extend(["--resume", str(root / name / "last.pt")])
                if allow_lr_change:
                    cmd.append("--allow-resume-lr-change")
                p = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True, timeout=90)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            run("full", 16)
            run("split", 8)
            run("split", 16, True)
            a = torch.load(root / "full" / "last.pt", weights_only=False)
            b = torch.load(root / "split" / "last.pt", weights_only=False)
            for group in ["model", "ema"]:
                for key in a[group]:
                    torch.testing.assert_close(a[group][key], b[group][key], rtol=0, atol=0)
            self.assertEqual(b["images_seen"], 16)
            self.assertEqual(len(b["rng"]), 2)
            self.assertFalse(torch.equal(b["rng"][0]["cpu"], b["rng"][1]["cpu"]))
            cfg["lr"] = 5e-5
            run("split", 24, True, True)
            c = torch.load(root / "split" / "last.pt", weights_only=False)
            self.assertEqual(c["step"], 6)
            self.assertEqual(c["images_seen"], 24)
            self.assertEqual(c["optimizer"]["param_groups"][0]["lr"], 5e-5)
            for state in c["optimizer"]["state"].values():
                self.assertEqual(state["step"].item(), 6)
            self.assertEqual(c["continuation"]["previous_lr"], 1e-4)
            self.assertEqual(c["continuation"]["images_seen"], 16)
            self.assertTrue(c["continuation"]["explicit_lr_change"])
            cfg["lr"] = 1e-4
            run("full", 24, True)
            d = torch.load(root / "full" / "last.pt", weights_only=False)
            for rank in range(2):
                torch.testing.assert_close(c["rng"][rank]["cpu"], d["rng"][rank]["cpu"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
