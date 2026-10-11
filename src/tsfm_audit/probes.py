"""Corruption probes: the core experimental manipulation.

All probes keep marginal distributions intact and break only the
target–covariate alignment. That is what makes the swap gap interpretable
as *use* rather than robustness-to-noise.
"""
from __future__ import annotations

import numpy as np


def swap_across_series(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Swap each series' covariates with another series' (derangement).

    Args:
        X: shape (n_series, time, n_cov). Past+future stacked; swap whole trajectories.
        rng: seeded generator for reproducibility.

    Returns:
        X_swapped with no series keeping its own covariates (when n>1).
    """
    n = X.shape[0]
    if n < 2:
        raise ValueError("need >=2 series to swap across series")
    perm = rng.permutation(n)
    # derange: re-roll until nobody keeps their own (cheap for benchmark n)
    tries = 0
    while np.any(perm == np.arange(n)) and tries < 100:
        perm = rng.permutation(n)
        tries += 1
    return X[perm]


def permute_time(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute each series' covariates along time (keeps marginals, breaks alignment)."""
    Xp = X.copy()
    for i in range(X.shape[0]):
        Xp[i] = X[i, rng.permutation(X.shape[1])]
    return Xp


def truncate_context(Y: np.ndarray, X: np.ndarray, keep_last: int) -> tuple[np.ndarray, np.ndarray]:
    """Context probe: keep only the last `keep_last` timesteps of history."""
    if keep_last <= 0 or keep_last > Y.shape[1]:
        raise ValueError(f"keep_last={keep_last} invalid for context {Y.shape[1]}")
    return Y[:, -keep_last:], X[:, -keep_last:, :]
