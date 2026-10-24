"""Shared image-index split used by pixel training and quality evaluation."""
import numpy as np


def split_indices(count, cfg):
    nval = cfg.get("val_count", 1000)
    if not 0 < nval < count:
        raise ValueError("Need nonempty training and validation splits")
    mode = cfg.get("split_mode", "random")
    if mode == "random":
        order = np.random.default_rng(cfg.get("seed", 42)).permutation(count)
    elif mode == "validation_first":
        order = np.arange(count)
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    return order[nval:], order[:nval]
