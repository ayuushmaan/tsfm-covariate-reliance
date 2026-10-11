"""Per-model adapters: wrap each checkpoint to predict(Y_past, X_past, X_future) -> forecast.

DESIGN RULE (local-machine friendly): importing this module must NEVER import
torch / chronos / fev or trigger any download. All heavy imports live INSIDE
the loader functions, which run only on the GPU box (marimo/Modal).

Status (Oct 2026, verified):
- chronos2: native past + known covariates via predict_df(df, future_df). HF amazon/chronos-2.
- bolt: univariate control via ChronosBoltPipeline.predict_df(df) — covariates
  are explicitly DROPPED, so its swap gap must be ~0. HF amazon/chronos-bolt-small.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .harness import PredictFn

CHRONOS2_ID = "amazon/chronos-2"
BOLT_ID = "amazon/chronos-bolt-small"  # 48M: cheapest faithful control
FEV_MINI_YAML = "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks_mini.yaml"


def naive_last_value(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray, horizon: int | None = None) -> np.ndarray:
    """Covariate-blind baseline: repeat last observation. Gap must be 0."""
    H = X_future.shape[1] if horizon is None else horizon
    last = Y_past[:, -1:]
    return np.repeat(last, H, axis=1)


def oracle_lagged(beta: float = 1.0, lag: int = 3, ar: float = 0.6):
    """Covariate-using oracle for planted demo: knows the true law. Gap tracks ceiling."""

    def predict(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        N, T = Y_past.shape
        H = X_future.shape[1]
        full_x = np.concatenate([X_past[:, :, 0], X_future[:, :, 0]], axis=1)
        out = np.zeros((N, H))
        y_last = Y_past[:, -1].copy()
        for h in range(H):
            x_lag = full_x[:, T + h - lag] if T + h - lag >= 0 else 0.0
            y_new = ar * y_last + beta * x_lag
            out[:, h] = y_new
            y_last = y_new
        return out

    predict.__name__ = f"oracle_lagged_beta{beta}"
    return predict


def arrays_to_long_df(Y_past, X_past, X_future, ids=None, past_ts=None, fut_ts=None,
                      col_names=None, known_names=None, cat_maps=None):
    """Convert (N,T)/(N,T,C)/(N,H,C) arrays to Chronos long-format (context_df, future_df).

    Pure pandas/numpy — no model, no download. Unit-tested locally.
    - col_names: real covariate names (default cov_0..); cat_maps: {col: [cats]}
      to decode integer codes back to strings (Chronos-2 encodes them itself).
    - future_df keeps only known_names (declared known-dynamic); other cols are
      past-only. Fallback: cols with any non-NaN future value.
    ids / past_ts / fut_ts optionally carry real benchmark stamps; otherwise
    a synthetic daily range is used (pipelines only need regularity).
    """
    import numpy as _np
    import pandas as pd

    N, T = Y_past.shape
    H = X_future.shape[1]
    C = X_past.shape[2]
    if ids is None:
        ids = [f"s{i}" for i in range(N)]
    if past_ts is None:
        base = pd.Timestamp("2020-01-01")
        past_ts = [pd.date_range(base, periods=T, freq="D")] * N
        fut_ts = [pd.date_range(base + pd.Timedelta(days=T), periods=H, freq="D")] * N
    cov_cols = list(col_names) if col_names else [f"cov_{c}" for c in range(C)]
    cat_maps = cat_maps or {}
    if known_names is None:
        fut_has = [bool(_np.any(~_np.isnan(X_future[:, :, c]))) for c in range(C)] if C else []
        known_set = {cc for cc, ok in zip(cov_cols, fut_has) if ok}
    else:
        known_set = set(known_names)

    def decode(col: str, vals: _np.ndarray):
        if col not in cat_maps:
            return vals.astype(float)
        cats = cat_maps[col]
        code = _np.where(_np.isnan(vals), 0, _np.round(vals)).astype(int)
        code = _np.clip(code, 0, len(cats) - 1)
        return pd.Series([cats[i] for i in code], dtype=object).to_numpy()

    ctx_parts, fut_parts = [], []
    for i in range(N):
        ctx = pd.DataFrame({"id": ids[i], "timestamp": pd.DatetimeIndex(past_ts[i]), "target": Y_past[i].astype(float)})
        fut = pd.DataFrame({"id": ids[i], "timestamp": pd.DatetimeIndex(fut_ts[i])})
        for c, cc in enumerate(cov_cols):
            ctx[cc] = decode(cc, X_past[i, :, c])
            if cc in known_set:
                fut[cc] = decode(cc, X_future[i, :, c])
        ctx_parts.append(ctx)
        fut_parts.append(fut)

    context_df = pd.concat(ctx_parts, ignore_index=True)
    future_df = pd.concat(fut_parts, ignore_index=True)
    return context_df, future_df, [cc for cc in cov_cols if cc in known_set]


def pred_df_to_array(pred_df: Any, n_series: int, horizon: int) -> np.ndarray:
    """Extract (N,H) point forecasts from a Chronos predict_df result.

    predict_df returns one row per (series, target, step) with a "predictions"
    column (mean). Sorted by (id, target_name, timestamp) upstream by pipeline.
    """
    vals = pred_df["predictions"].to_numpy(dtype=float)
    return vals.reshape(n_series, horizon)


def make_chronos2_predict(pipeline: Any, batch_size: int = 64, ids=None, past_ts=None, fut_ts=None,
                          col_names=None, known_names=None, cat_maps=None) -> PredictFn:
    """Wrap a loaded Chronos2Pipeline -> PredictFn. Pipeline must already be on GPU box."""

    def predict(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        context_df, future_df, _ = arrays_to_long_df(
            Y_past, X_past, X_future, ids, past_ts, fut_ts, col_names, known_names, cat_maps)
        out = pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=X_future.shape[1],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            batch_size=batch_size,
        )
        return pred_df_to_array(out, Y_past.shape[0], X_future.shape[1])

    predict.__name__ = "chronos2"
    return predict


def make_bolt_predict(pipeline: Any, batch_size: int = 256, ids=None, past_ts=None, fut_ts=None,
                      col_names=None, known_names=None, cat_maps=None) -> PredictFn:
    """Wrap a loaded ChronosBoltPipeline -> PredictFn. Covariates DROPPED (control)."""

    def predict(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        context_df, _, _ = arrays_to_long_df(
            Y_past, X_past, X_future, ids, past_ts, fut_ts, col_names, known_names, cat_maps)
        ctx_target_only = context_df[["id", "timestamp", "target"]]
        out = pipeline.predict_df(
            ctx_target_only,
            prediction_length=X_future.shape[1],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            batch_size=batch_size,
        )
        return pred_df_to_array(out, Y_past.shape[0], X_future.shape[1])

    predict.__name__ = "bolt_control"
    return predict


def load_chronos2(device_map: str = "cuda"):
    """Heavy loader — call ONLY on the GPU box. Imports chronos + downloads 120M ckpt."""
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(CHRONOS2_ID, device_map=device_map)


def load_bolt(device_map: str = "cuda"):
    """Heavy loader — call ONLY on the GPU box. Downloads ~48M ckpt."""
    from chronos import ChronosBoltPipeline

    return ChronosBoltPipeline.from_pretrained(BOLT_ID, device_map=device_map)
