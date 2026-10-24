import unittest

import torch

from sample_pixel import AutoGuidedPredictor


class Predictor(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return torch.full_like(x, self.value)


class AutoGuidanceTest(unittest.TestCase):
    def test_scale_one_is_exact_and_skips_guide(self):
        good, bad = Predictor(3), Predictor(1)
        x, t = torch.randn(2, 3, 4, 4), torch.ones(2)
        out = AutoGuidedPredictor(good, bad, 1)(x, t)
        self.assertTrue(torch.equal(out, torch.full_like(x, 3)))
        self.assertEqual(bad.calls, 0)

    def test_clean_prediction_extrapolation(self):
        x, t = torch.randn(2, 3, 4, 4), torch.ones(2)
        out = AutoGuidedPredictor(Predictor(3), Predictor(1), 1.5)(x, t)
        self.assertTrue(torch.equal(out, torch.full_like(x, 4)))

    def test_interval_leaves_late_denoising_unchanged(self):
        x, t = torch.zeros(2, 3, 4, 4), torch.tensor([.5, .9])
        out = AutoGuidedPredictor(Predictor(3), Predictor(1), 1.5, .1, .8)(x, t)
        self.assertTrue(torch.equal(out[0], torch.full_like(x[0], 4)))
        self.assertTrue(torch.equal(out[1], torch.full_like(x[1], 3)))
