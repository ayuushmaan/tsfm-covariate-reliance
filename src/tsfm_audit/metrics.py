"""Metrics: MASE, WQL, skill scores — matching fev-bench conventions."""
from __future__ import annotations

import numpy as np


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1) -> float:
    """Mean Absolute Scaled Error. Lower is better. Ignores NaN truth steps.

    Returns NaN for unmeasurable series (seasonal-naive scale ~0 relative to
    series magnitude, e.g. intermittent all-zero sales): callers average with
    nanmean so one degenerate series cannot dominate a task mean.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid):
        return float("nan")
    scale_src = y_train[~np.isnan(y_train)]
    if len(scale_src) <= seasonality:
        return float("nan")
    naive_scale = np.mean(np.abs(scale_src[seasonality:] - scale_src[:-seasonality]))
    if naive_scale <= 1e-8 * (np.mean(np.abs(scale_src)) + 1e-8):
        return float("nan")
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])) / naive_scale)


def wql(y_true: np.ndarray, quantiles: np.ndarray, q_levels: np.ndarray) -> float:
    """Weighted Quantile Loss over quantile forecasts. Lower is better.

    Args:
        y_true: (H,), quantiles: (H, Q), q_levels: (Q,) in (0,1).
    """
    y_true = np.asarray(y_true, float)[:, None]
    quantiles = np.asarray(quantiles, float)
    q = np.asarray(q_levels, float)[None, :]
    pinball = np.maximum(q * (y_true - quantiles), (1 - q) * (quantiles - y_true))
    denom = np.mean(np.abs(y_true))
    if denom == 0:
        denom = 1e-8
    return float(np.mean(pinball) / denom)


def skill_score(loss_model: float, loss_baseline: float) -> float:
    """Skill vs baseline: 1 - loss_model/loss_baseline. Higher is better."""
    if loss_baseline == 0:
        return 0.0
    return float(1.0 - loss_model / loss_baseline)
