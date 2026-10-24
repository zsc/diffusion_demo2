"""Score saved clean Inception features with a stable symmetric FID formula."""
import argparse
import json
from pathlib import Path

import numpy as np


def psd_sqrt(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) * .5)
    if values.min() < -1e-7 * max(1, values.max()):
        raise ValueError("Covariance is not positive semidefinite")
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T


def stable_fid(real, fake):
    """Same Gaussian Wasserstein distance; avoid sqrtm of nonsymmetric A @ B.

    With N < feature dimension the covariance is singular. Symmetrizing the
    PSD sandwich avoids large spurious complex parts from a general sqrtm.
    """
    a, b = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = psd_sqrt(ca)
    middle = root @ cb @ root
    eigenvalues = np.linalg.eigvalsh((middle + middle.T) * .5)
    if eigenvalues.min() < -1e-7 * max(1, eigenvalues.max()):
        raise ValueError("Covariance sandwich is not positive semidefinite")
    fid = np.square(a.mean(0) - b.mean(0)).sum() + np.trace(ca) + np.trace(cb)
    fid -= 2 * np.sqrt(np.maximum(eigenvalues, 0)).sum()
    if not np.isfinite(fid) or fid < -1e-4:
        raise ValueError(f"Invalid FID {fid}")
    return float(max(0, fid))


def score_features(real, fake):
    from cleanfid.fid import kernel_distance
    if not (np.isfinite(real).all() and np.isfinite(fake).all()):
        raise ValueError("Nonfinite Inception features")
    np.random.seed(12345)
    parts = [kernel_distance(real, fake, num_subsets=1, max_subset_size=100) for _ in range(100)]
    return dict(clean_fid=stable_fid(real, fake), kid_mean=float(np.mean(parts)),
                kid_subset_std=float(np.std(parts)), kid_subsets=100,
                kid_subset_size=min(100, len(real), len(fake)),
                real_count=len(real), fake_count=len(fake),
                fid_solver="float64 symmetric PSD eigendecomposition")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    p = Path(args.directory)
    result = score_features(np.load(p / "real_features.npy"), np.load(p / "fake_features.npy"))
    (p / "feature_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
