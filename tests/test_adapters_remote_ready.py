"""Adapter wiring tests — mocks only, zero downloads.

Guards the local-machine rule: importing adapters must not pull torch/chronos/fev.
"""
import sys

import numpy as np
import pandas as pd

from tsfm_audit.adapters import (
    arrays_to_long_df,
    make_bolt_predict,
    make_chronos2_predict,
    pred_df_to_array,
)
from tsfm_audit.harness import run_probe
from tsfm_audit.planted import PlantedConfig, generate_planted_panel


class FakeCovariatePipeline:
    """Follows the planted law through the DataFrame interface — uses X, so swap must hurt."""

    def __init__(self, beta: float = 1.0, lag: int = 3, ar: float = 0.6):
        self.beta, self.lag, self.ar = beta, lag, ar

    def predict_df(self, df, future_df=None, prediction_length=None, **kw):
        rows = []
        for sid, g in df.groupby("id"):
            g = g.sort_values("timestamp")
            f = future_df[future_df["id"] == sid].sort_values("timestamp")
            full_x = np.concatenate([g["cov_0"].to_numpy(), f["cov_0"].to_numpy()])
            T = len(g)
            y_last = float(g["target"].iloc[-1])
            for h in range(prediction_length):
                x_lag = full_x[T + h - self.lag] if T + h - self.lag >= 0 else 0.0
                y_last = self.ar * y_last + self.beta * x_lag
                rows.append({"id": sid, "timestamp": f["timestamp"].iloc[h], "target_name": "target", "predictions": y_last})
        return pd.DataFrame(rows)


class FakeBoltPipeline:
    def predict_df(self, df, prediction_length=None, **kw):
        rows = []
        for sid, g in df.groupby("id"):
            last = float(g["target"].iloc[-1])
            ts = pd.date_range("2020-01-01", periods=len(g) + prediction_length, freq="D")[-prediction_length:]
            for t in ts:
                rows.append({"id": sid, "timestamp": t, "target_name": "target", "predictions": last})
        return pd.DataFrame(rows)


def test_import_is_light():
    for mod in ("torch", "chronos", "fev"):
        assert mod not in sys.modules, f"{mod} imported at module load — must stay lazy"


def test_long_df_roundtrip():
    rng = np.random.default_rng(0)
    Y, Xp, Xf = rng.normal(size=(4, 10)), rng.normal(size=(4, 10, 2)), rng.normal(size=(4, 5, 2))
    ctx, fut, covs = arrays_to_long_df(Y, Xp, Xf)
    assert ctx.shape[0] == 4 * 10 and fut.shape[0] == 4 * 5
    assert covs == ["cov_0", "cov_1"]
    assert set(["id", "timestamp", "target"]) <= set(ctx.columns)


def test_chronos2_wrapper_uses_covariates():
    panel = generate_planted_panel(1.0, PlantedConfig(n_series=8, seed=0))
    r = run_probe(
        "c2", "t", panel["Y_past"], panel["X_past"], panel["X_future"], panel["Y_future"],
        make_chronos2_predict(FakeCovariatePipeline()), seed=0,
    )
    assert r.swap_gap > 0.01


def test_bolt_wrapper_ignores_covariates():
    panel = generate_planted_panel(1.0, PlantedConfig(n_series=8, seed=0))
    r = run_probe(
        "bolt", "t", panel["Y_past"], panel["X_past"], panel["X_future"], panel["Y_future"],
        make_bolt_predict(FakeBoltPipeline()), seed=0,
    )
    assert abs(r.swap_gap) < 1e-9


def test_pred_df_to_array_shape():
    df = pd.DataFrame({"predictions": np.arange(6.0)})
    assert pred_df_to_array(df, 2, 3).shape == (2, 3)
