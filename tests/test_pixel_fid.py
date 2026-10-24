import unittest

import numpy as np

from scripts.score_pixel_features import stable_fid


class StableFIDTest(unittest.TestCase):
    def test_singular_identity(self):
        x = np.random.default_rng(42).normal(size=(8, 32))
        self.assertAlmostEqual(stable_fid(x, x[::-1]), 0, places=4)

    def test_gaussian_shift_scale(self):
        x = np.random.default_rng(17).normal(size=(128, 8))
        y = 2 * x + 3
        expected = np.square(x.mean(0) - y.mean(0)).sum() + np.trace(np.cov(x, rowvar=False))
        self.assertAlmostEqual(stable_fid(x, y), expected, places=7)
        self.assertAlmostEqual(stable_fid(y, x), expected, places=7)
