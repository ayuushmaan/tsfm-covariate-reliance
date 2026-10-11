"""Probe harness: load task -> predict intact -> predict corrupted -> score.

Model-agnostic: a model is any callable
    predict(Y_past (N,T), X_past (N,T,C), X_future (N,H,C)) -> (N,H) point forecast.
Adapters in adapters.py wrap each checkpoint to this signature.
Results bank per model as JSONL so a box death costs only the missing model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .metrics import mase
from .probes import permute_time, swap_across_series

PredictFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class ProbeResult:
    model: str
    task: str
    seed: int
    mase_intact: float
    mase_swapped: float
    mase_timeperm: float
    swap_gap: float  # mase_swapped - mase_intact; >0 means reliance
    timeperm_gap: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_probe(
    model_name: str,
    task_name: str,
    Y_past: np.ndarray,
    X_past: np.ndarray,
    X_future: np.ndarray,
    Y_future: np.ndarray,
    predict: PredictFn,
    seed: int = 0,
    seasonality: int = 1,
) -> ProbeResult:
    """Run intact + 2 corruptions for one (model, task, seed)."""
    rng = np.random.default_rng(seed)
    H = Y_future.shape[1]

    X_full = np.concatenate([X_past, X_future], axis=1)  # (N, T+H, C)
    X_swap = swap_across_series(X_full, rng)
    X_perm = permute_time(X_full, rng)

    def split(Xf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = Y_past.shape[1]
        return Xf[:, :T, :], Xf[:, T : T + H, :]

    pred_intact = predict(Y_past, X_past, X_future)
    Xp_s, Xf_s = split(X_swap)
    Xp_p, Xf_p = split(X_perm)
    pred_swap = predict(Y_past, Xp_s, Xf_s)
    pred_perm = predict(Y_past, Xp_p, Xf_p)

    def mean_mase(pred: np.ndarray) -> float:
        vals = [mase(Y_future[i], pred[i], Y_past[i], seasonality) for i in range(len(Y_past))]
        return float(np.nanmean(vals))

    mi, ms, mp = mean_mase(pred_intact), mean_mase(pred_swap), mean_mase(pred_perm)
    return ProbeResult(model_name, task_name, seed, mi, ms, mp, ms - mi, mp - mi)


def run_probe_windows(
    model_name: str,
    task_name: str,
    windows: list[dict],
    predict: PredictFn,
    seed: int = 0,
    seasonality: int = 1,
    max_context: int | None = None,
) -> ProbeResult:
    """Cross-window probe for single-series tasks.

    Each window is one (Y_past, X_past, X_future, Y_future) dict. Covariate
    trajectories are swapped ACROSS windows (derangement): same idea as the
    cross-series swap, but the donor pool is time rather than sibling series.
    Caveat for the paper: windows come from different periods, so marginals
    match only approximately — the planted dose-response bounds this concern.

    Windows are truncated to a common tail length (min length, max_context).
    """
    if len(windows) < 2:
        raise ValueError("need >=2 windows to swap across windows")
    H0 = windows[0]["Y_future"].shape[0]
    C0 = windows[0]["X_past"].shape[1]
    for w in windows:
        if w["Y_future"].shape[0] != H0 or w["X_past"].shape[1] != C0:
            raise ValueError("windows must share horizon and covariate count")
    T_use = min(w["Y_past"].shape[0] for w in windows)
    if max_context is not None:
        T_use = min(T_use, max_context)

    Y = np.stack([w["Y_past"][-T_use:] for w in windows])          # (W, T)
    Xp = np.stack([w["X_past"][-T_use:, :] for w in windows])      # (W, T, C)
    Xf = np.stack([w["X_future"] for w in windows])                # (W, H, C)
    Yf = np.stack([w["Y_future"] for w in windows])                # (W, H)

    rng = np.random.default_rng(seed)
    X_full = np.concatenate([Xp, Xf], axis=1)
    X_swap = swap_across_series(X_full, rng)
    X_perm = permute_time(X_full, rng)

    def split(Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return Xc[:, :T_use, :], Xc[:, T_use : T_use + H0, :]

    results = {}
    for name, Xc in [("intact", X_full), ("swapped", X_swap), ("timeperm", X_perm)]:
        Xpc, Xfc = split(Xc)
        preds = np.stack([
            predict(Y[i : i + 1], Xpc[i : i + 1], Xfc[i : i + 1])[0] for i in range(len(windows))
        ])
        results[name] = float(np.nanmean([
            mase(Yf[i], preds[i], Y[i], seasonality) for i in range(len(windows))
        ]))
    mi, ms, mp = results["intact"], results["swapped"], results["timeperm"]
    return ProbeResult(model_name, task_name, seed, mi, ms, mp, ms - mi, mp - mi)
