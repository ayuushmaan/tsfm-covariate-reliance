"""Planted-signal generator: target = own dynamics + beta * lagged covariate.

Gives a dose-response curve per model: the beta at which covariate use appears.
Runs on the fly — no downloads, no GPU — so the harness is testable tonight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlantedConfig:
    n_series: int = 32
    context: int = 128
    horizon: int = 24
    lag: int = 3
    ar_coef: float = 0.6
    noise: float = 0.2
    seed: int = 0


def generate_planted_panel(beta: float, cfg: PlantedConfig) -> dict:
    """Generate one panel with a known covariate effect size beta.

    Returns dict with Y_past (N,T), X_past (N,T,1), X_future (N,H,1), Y_future (N,H).
    Covariate x ~ AR(1); target y_t = ar*y_{t-1} + beta*x_{t-lag} + noise.
    Future x is known (the informative case); future y must be forecast.
    """
    rng = np.random.default_rng(cfg.seed + int(beta * 1000))
    N, T, H = cfg.n_series, cfg.context, cfg.horizon
    total = T + H
    x = np.zeros((N, total))
    y = np.zeros((N, total))
    x[:, 0] = rng.normal(size=N)
    y[:, 0] = rng.normal(size=N)
    for t in range(1, total):
        x[:, t] = 0.7 * x[:, t - 1] + rng.normal(size=N)
        lagged = x[:, t - cfg.lag] if t - cfg.lag >= 0 else 0.0
        y[:, t] = cfg.ar_coef * y[:, t - 1] + beta * lagged + cfg.noise * rng.normal(size=N)
    return {
        "Y_past": y[:, :T],
        "X_past": x[:, :T, None],
        "X_future": x[:, T:, None],
        "Y_future": y[:, T:],
        "beta": beta,
    }
