"""Ceiling v3: DIRECT multi-output ridge (no iteration, no divergence).

Per-series ridge mapping fixed history features (own lags + covariate lags,
incl. seasonal 24/168 when history allows) directly to all H future steps.
Fit on sliding windows of the past; applied once at the cutoff.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV


def _ffill_path(a: np.ndarray) -> np.ndarray:
    out = a.copy().astype(float)
    for t in range(1, out.shape[0]):
        prev, cur = out[t - 1], out[t]
        out[t] = np.where(np.isnan(cur), prev, cur)
    if np.isnan(out).any():
        col_med = np.nanmedian(out, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        out = np.where(np.isnan(out), col_med, out)
    return out


def ceiling_ridge(
    Y_past: np.ndarray,
    X_past: np.ndarray,
    X_future: np.ndarray,
    max_lags: int = 4,
    skip_cols: tuple = (),
) -> np.ndarray:
    N, T = Y_past.shape
    H = X_future.shape[1]
    C = X_past.shape[2] if X_past.ndim == 3 else 0
    if C == 0:
        return np.repeat(Y_past[:, -1:], H, axis=1)
    use = [c for c in range(C) if c not in set(skip_cols)]
    if not use:
        return np.repeat(Y_past[:, -1:], H, axis=1)
    K = max(1, min(max_lags, T // 8))
    if T < 64:
        K = 1
    lags = list(range(1, K + 1))
    L = max(lags)
    preds = np.zeros((N, H))
    for i in range(N):
        if T - H - L < 8:
            preds[i] = Y_past[i, -1]
            continue
        Xp = _ffill_path(X_past[i][:, use])
        y = _ffill_path(Y_past[i][:, None])[:, 0]
        if T - H - L < 8:
            preds[i] = y[-1]
            continue
        # per-horizon designs: y-lags stay at the cutoff, x-lags slide with the
        # target step so KNOWN future covariates enter as features. One RidgeCV
        # per step (a shared alpha is dominated by noisy far horizons).
        Xd_base = np.stack([[y[t - k] for k in lags] for t in range(L, T - H)], axis=0)
        full_x = _ffill_path(np.concatenate([Xp, X_future[i][:, use]], axis=0))
        for h in range(H):
            Xx = np.stack([np.concatenate([full_x[t + h - k] for k in lags]) for t in range(L, T - H)], axis=0)
            Xd = np.concatenate([Xd_base, Xx], axis=1)
            yd = np.stack([y[t + h] for t in range(L, T - H)], axis=0)
            model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(Xd, yd)
            feat = np.concatenate([[y[T - k] for k in lags],
                                   np.concatenate([full_x[T + h - k] for k in lags])]).reshape(1, -1)
            preds[i, h] = model.predict(feat)[0]
    return preds
