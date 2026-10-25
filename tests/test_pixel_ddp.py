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
from pixel_ddp_experiment import (validate_resume_config, training_time_parameters,
                                  training_time_sampling, sample_training_times, training_lr)


class PixelDDPTests(unittest.TestCase):
    def test_batch_and_lr_schedule_changes_are_explicit(self):
        previous = dict(global_batch=1024, lr=5e-5, warmup_images=100)
        with self.assertRaisesRegex(ValueError, "global_batch"):
            validate_resume_config(previous, {**previous, "global_batch": 2048})
        validate_resume_config(previous, {**previous, "global_batch": 2048}, allow_batch_change=True)
        for value in [0, -1, 2.5]:
            with self.assertRaises(ValueError):
                validate_resume_config(previous, {**previous, "global_batch": value}, allow_batch_change=True)
        cfg = {**previous, "lr_schedule": "cosine", "lr_decay_start_images": 100,
               "lr_decay_end_images": 200, "lr_min_ratio": .2}
        with self.assertRaisesRegex(ValueError, "LR schedule"):
            validate_resume_config(previous, cfg)
        validate_resume_config(previous, cfg, allow_lr_change=True)
        self.assertAlmostEqual(training_lr(cfg, 100), 5e-5)
        self.assertAlmostEqual(training_lr(cfg, 150), 3e-5)
        self.assertAlmostEqual(training_lr(cfg, 200), 1e-5)
        self.assertAlmostEqual(training_lr(cfg, 300), 1e-5)
        self.assertAlmostEqual(training_lr(previous, 50), 2.5e-5)

    def test_stratified_time_sampling_preserves_cdf_coverage_and_rng(self):
        torch.manual_seed(19)
        state = torch.get_rng_state()
        times = sample_training_times(64, "cpu", -.2, .8, "stratified")
        z = (torch.logit(times) + .2) / .8
        quantiles = .5 * (1 + torch.erf(z / (2 ** .5)))
        torch.testing.assert_close(torch.floor(quantiles * 64), torch.arange(64).float())
        self.assertTrue(times.isfinite().all())
        self.assertFalse(torch.equal(times, sample_training_times(64, "cpu", -.2, .8, "stratified")))
        torch.set_rng_state(state)
        torch.testing.assert_close(times, sample_training_times(64, "cpu", -.2, .8, "stratified"), rtol=0, atol=0)
        torch.set_rng_state(state)
        expected = (torch.randn(64) * .8 - .2).sigmoid()
        torch.set_rng_state(state)
        torch.testing.assert_close(expected, sample_training_times(64, "cpu", -.2, .8), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "time distribution"):
            validate_resume_config({}, {"time_sampling": "stratified"})
        validate_resume_config({}, {"time_sampling": "stratified"}, allow_time_change=True)
        with self.assertRaises(ValueError):
            training_time_sampling({"time_sampling": "invalid"})

    def test_time_change_is_explicit_and_validated(self):
        self.assertEqual(training_time_parameters({}), (-.8, .8))
        validate_resume_config({}, {"time_logit_mean": -.8, "time_logit_std": .8})
        with self.assertRaisesRegex(ValueError, "time distribution"):
            validate_resume_config({}, {"time_logit_mean": -.4})
        validate_resume_config({}, {"time_logit_mean": -.4}, allow_time_change=True)
        for key, value in [("time_logit_mean", float("nan")), ("time_logit_std", 0), ("time_logit_std", float("inf"))]:
            with self.assertRaises(ValueError):
                validate_resume_config({}, {key: value}, allow_time_change=True)

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
            cfg["extra_ema_decays"] = [.95, .999]
            np.save(root / "data.npy", np.random.default_rng(7).integers(0, 256, (10, 48, 48, 3), dtype=np.uint8))
            def run(name, images, resume=False, allow_lr_change=False, allow_batch_change=False):
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
                if allow_batch_change:
                    cmd.append("--allow-resume-batch-change")
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
            for decay in a["extra_emas"]:
                for key in a["extra_emas"][decay]:
                    torch.testing.assert_close(a["extra_emas"][decay][key], b["extra_emas"][decay][key], rtol=0, atol=0)
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
            cfg["time_sampling"] = "stratified"
            run("strat_full", 16)
            run("strat_split", 8)
            run("strat_split", 16, True)
            a = torch.load(root / "strat_full" / "last.pt", weights_only=False)
            b = torch.load(root / "strat_split" / "last.pt", weights_only=False)
            for group in ["model", "ema"]:
                for key in a[group]:
                    torch.testing.assert_close(a[group][key], b[group][key], rtol=0, atol=0)
            for rank in range(2):
                torch.testing.assert_close(a["rng"][rank]["cpu"], b["rng"][rank]["cpu"], rtol=0, atol=0)
            cfg["global_batch"] = 8
            run("strat_split", 24, True, allow_batch_change=True)
            changed = torch.load(root / "strat_split" / "last.pt", weights_only=False)
            self.assertEqual(changed["step"], 5)
            self.assertTrue(changed["continuation"]["explicit_batch_change"])
            decay = min(cfg["ema_reference_decay"], 7 / 16) ** 2
            for key in changed["ema"]:
                torch.testing.assert_close(changed["ema"][key], b["ema"][key] * decay + changed["model"][key] * (1-decay))


if __name__ == "__main__":
    unittest.main()
