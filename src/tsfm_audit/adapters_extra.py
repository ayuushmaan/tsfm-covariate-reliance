"""Extra model adapters: TiRex, t0, Toto-1.0, TimesFM-3.0/2.5, Moirai-2.0, TabPFN-TS, Chronos-2-synth.

Same DESIGN RULE as adapters.py: importing this module must NEVER import
heavy third-party packages or trigger downloads. Everything heavy lives
INSIDE loader functions, which run only on the GPU box.

Covariate-support summary (Oct 2026, verified from docs; validated live on box):
- chronos2_synth: identical API to Chronos-2 (same adapter shape).
- t0 (alpha 102M, Apache-2.0): native past + known-future covariates.
  context (B, V, T) carries target + past covs as variates; known covs go in
  future_covariates (B, F, T+H). String cats -> float codes (t0 is numeric).
- toto1 (151M, Apache-2.0): multivariate input; covariates as extra channels,
  target = channel 0, point forecast = sample median.
- timesfm3 (330M): native past_only (F, T) + past_future (F, T+H) covariates.
- timesfm25: covariates via XReg in-context linear regression (different
  mechanism, still probe-able: swap must hurt if the regression uses X).
- tirex (35M xLSTM): univariate, no covariate support -> control (like Bolt).
- moirai2 (small): univariate by design (covariate support dropped in 2.0) ->
  control. GluonTS predictor API, per-series loop.
- tabpfn_ts: known-future covariates only (past/static dropped by the lib).
  predict_df with item_id/timestamp/target schema. LOCAL mode may require a
  license acceptance on first run; CLIENT mode needs an account token.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .harness import PredictFn

TIREX_ID = "NX-AI/TiRex"
T0_ID = "theforecastingcompany/t0-alpha"
TOTO1_ID = "Datadog/Toto-Open-Base-1.0"
TIMESFM3_ID = "google/timesfm-3.0-pytorch"
TIMESFM25_ID = "google/timesfm-2.5-200m-pytorch"
MOIRAI2_ID = "Salesforce/moirai-2.0-R-small"
CHRONOS2_SYNTH_ID = "autogluon/chronos-2-synth"


def _tonp(t: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(t, torch.Tensor):
            t = t.detach().cpu()
    except ImportError:
        pass
    return np.asarray(t, dtype=float)


def _as_float(x: np.ndarray) -> np.ndarray:
    return _tonp(x)


# ---------------------------------------------------------------- TiRex (control)

def make_tirex_predict(model: Any) -> PredictFn:
    """Wrap a loaded TiRex ForecastModel. Ignores X (no covariate support)."""
    import torch

    def predict(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        H = X_future.shape[1]
        with torch.no_grad():
            _, mean = model.forecast(context=torch.as_tensor(Y_past, dtype=torch.float32), prediction_length=H)
        return np.asarray(mean, dtype=float)

    predict.__name__ = "tirex_control"
    return predict


def load_tirex(device: str = "cuda"):
    from tirex import load_model

    model = load_model(TIREX_ID)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


# ---------------------------------------------------------------- t0 (native covariates)

def make_t0_predict(model: Any, n_known: int = 0) -> PredictFn:
    """Wrap T0Forecaster. First n_known channels of X are known-future covs
    (passed as future_covariates over T+H); the rest are past covs passed as
    extra context variates. Point forecast = median of target variate."""
    import torch

    def predict(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        T, H = Y_past.shape[1], X_future.shape[1]
        Yt = torch.as_tensor(Y_past[:, None, :], dtype=torch.float32)
        out = []
        for i in range(Y_past.shape[0]):
            past_covs = ([X_past[i, :, n_known:].T] if X_past.shape[2] > n_known else [])
            ctx = torch.cat([Yt[i : i + 1]] + [torch.as_tensor(c[None], dtype=torch.float32) for c in past_covs], dim=1)
            kw = {}
            if n_known > 0:
                full_known = np.concatenate([X_past[i, :, :n_known], X_future[i, :, :n_known]], axis=0)  # (T+H, F)
                kw["future_covariates"] = torch.as_tensor(full_known.T[None], dtype=torch.float32)
            res = model.predict(ctx, horizon=H, quantile_levels=[0.5], **kw)
            med = _tonp(res.median if hasattr(res, "median") else res.quantiles[..., 0])
            med = med[:, 0, :] if med.ndim == 3 else med  # target variate 0 -> (1, H); batch is 1
            out.append(med[0])
        return np.stack(out)

    predict.__name__ = "t0"
    return predict


def load_t0(device: str = "cuda"):
    from t0 import T0Forecaster

    model = T0Forecaster.from_pretrained(T0_ID).eval()
    if hasattr(model, "to"):
        model = model.to(device)
    return model


# ---------------------------------------------------------------- Toto 1.0 (covariates as channels)

def make_toto_predict(forecaster: Any, device: str = "cuda", num_samples: int = 64) -> PredictFn:
    """Wrap TotoForecaster. Input (1+C, T) multivariate; target = channel 0.

    Drops zero-variance past channels (e.g. never-active shutdown flags):
    constants carry no information, and they break Toto's distribution head.
    The mask uses pooled (series, time) variance, which swaps/permutations
    preserve exactly, so intact and corrupted runs stay comparable.
    """
    import torch

    def predict(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        N, T = Y_past.shape
        H = X_future.shape[1]
        if X_past.shape[2]:
            keep = [c for c in range(X_past.shape[2]) if float(np.std(X_past[:, :, c])) > 1e-9]
            Xp = X_past[:, :, keep] if keep else np.zeros((N, T, 0))
        else:
            Xp = X_past
        out = []
        for i in range(N):
            series = np.concatenate([Y_past[i : i + 1], Xp[i].T], axis=0)  # (1+C', T)
            V = series.shape[0]
            ts = torch.as_tensor(series[None], dtype=torch.float32, device=device)  # (1, V, T)
            from toto.data.util.dataset import MaskedTimeseries

            inputs = MaskedTimeseries(
                series=ts,
                padding_mask=torch.ones_like(ts, dtype=torch.bool),
                id_mask=torch.zeros_like(ts),
                timestamp_seconds=torch.zeros_like(ts),
                time_interval_seconds=torch.full((1, V), 60.0, device=device),
            )
            fc = forecaster.forecast(inputs, prediction_length=H, num_samples=num_samples,
                                     samples_per_batch=num_samples)
            m = _tonp(fc.median)
            while m.ndim > 2:
                m = m[0]
            m = m[0, :H] if m.ndim == 2 else m[:H]  # target channel 0
            out.append(m)
        return np.stack(out)

    predict.__name__ = "toto1"
    return predict


def load_toto(device: str = "cuda"):
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto

    # Upstream bug: from_pretrained() never forwards the required
    # proxies/resume_download kwargs to _from_pretrained(). Patch once
    # (guarded: re-patching a patched classmethod recurses).
    if not getattr(Toto._from_pretrained, "_tsfm_ok", False):
        _orig = Toto._from_pretrained.__func__

        def _patched(cls, *, proxies=None, resume_download=False, **kw):
            return _orig(cls, proxies=proxies, resume_download=resume_download, **kw)

        _patched._tsfm_ok = True
        Toto._from_pretrained = classmethod(_patched)
    toto = Toto.from_pretrained(TOTO1_ID).to(device)
    return TotoForecaster(toto.model)


# ---------------------------------------------------------------- TimesFM 3.0 (native covariates)

def make_timesfm3_predict(forecaster: Any, n_known: int = 0) -> PredictFn:
    """Wrap TimesFM3 evaluator. Channels [n_known:] are past-only,
    channels [:n_known] are past-future (concat past+future). Single target."""

    def predict(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        N, T = Y_past.shape
        H = X_future.shape[1]
        contexts, po_list, pf_list = [], [], []
        for i in range(N):
            contexts.append(_as_float(Y_past[i])[None, :])
            po_list.append(_as_float(X_past[i, :, n_known:].T) if X_past.shape[2] > n_known else None)
            if n_known > 0:
                pf_list.append(np.concatenate([X_past[i, :, :n_known], X_future[i, :, :n_known]], axis=0).T)
            else:
                pf_list.append(None)
        outs = list(forecaster.predict_batch(
            contexts=contexts, horizon=H, past_only_covariates=po_list,
            past_future_covariates=pf_list, return_quantiles=False))
        return np.stack([np.asarray(o.forecast, dtype=float).reshape(-1)[:H] for o in outs])

    predict.__name__ = "timesfm3"
    return predict


def load_timesfm3(device: str = "cuda"):
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    config = ModelConfig(checkpoint_path=TIMESFM3_ID, per_core_batch_size=32, device=device)
    return TimesFM3Evaluator(config)


# ---------------------------------------------------------------- TimesFM 2.5 via XReg

def make_timesfm25_predict(model: Any, n_known: int = 0) -> PredictFn:
    """Wrap TimesFM-2.5 forecast_with_covariates (in-context regression).
    Only known channels are passed (XReg needs future values)."""

    def predict(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        N, T = Y_past.shape
        H = X_future.shape[1]
        if n_known > 0:
            inputs = [_as_float(Y_past[i]) for i in range(N)]
            dyn = {f"cov_{c}": [np.concatenate([X_past[j, :, c], X_future[j, :, c]])
                                for j in range(N)]
                   for c in range(n_known)}
            _, q = model.forecast_with_covariates(inputs=inputs, dynamic_numerical_covariates=dyn)
            q = _tonp(q)
            if q.ndim == 3:
                return q[:, :, q.shape[-1] // 2][:, :H]
            return q.reshape(N, H)
        out = []
        for i in range(N):
            point, _ = model.forecast(inputs=[_as_float(Y_past[i])], horizon=H)
            out.append(_tonp(point).reshape(-1)[:H])
        return np.stack(out)

    predict.__name__ = "timesfm25"
    return predict


def load_timesfm25(device: str = "cuda"):
    import timesfm  # pip install 'timesfm[torch]'

    cls = timesfm.TimesFM_2p5_200M_torch
    if hasattr(cls, "from_pretrained"):
        model = cls.from_pretrained(TIMESFM25_ID)
    else:
        model = cls()
    if hasattr(model, "compile"):
        try:
            from timesfm.configs import ForecastConfig

            model.compile(ForecastConfig(max_context=2048, max_horizon=256, per_core_batch_size=32,
                                           return_backcast=True))
        except Exception:
            pass
    return model


# ---------------------------------------------------------------- Moirai 2.0 (control)

def make_moirai2_predict(predictor: Any, freq: str = "D") -> PredictFn:
    """Wrap a Moirai2 GluonTS predictor. Ignores X (no covariate support in 2.0).
    Point forecast = 0.5 quantile (NOT the mean over quantile levels)."""
    from gluonts.dataset.common import ListDataset

    def _point(f: Any, H: int) -> np.ndarray:
        try:
            keys = [str(k) for k in f.forecast_keys]
            arr = np.asarray(f.forecast_array, dtype=float)
            return arr[keys.index("0.5")][:H]
        except Exception:
            return np.asarray(f.mean, dtype=float)[:H]

    def predict(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        import pandas as pd

        N, T = Y_past.shape
        H = X_future.shape[1]
        ds = ListDataset(
            [{"start": pd.Timestamp("2020-01-01"), "target": _as_float(Y_past[i])} for i in range(N)],
            freq=freq,
        )
        fcs = list(predictor.predict(ds))
        return np.stack([_point(f, H) for f in fcs])

    predict.__name__ = "moirai2_control"
    return predict


def load_moirai2(prediction_length: int = 24, context_length: int = 512):
    # Long-lived kernels may hold stale gluonts/uni2ts modules from a
    # different gluonts version (their singledispatch registries then miss).
    # Purge both namespaces so everything below imports consistently.
    import sys as _sys

    for _m in [m for m in _sys.modules if m.split(".")[0] in ("gluonts", "uni2ts")]:
        del _sys.modules[_m]
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(MOIRAI2_ID),
        prediction_length=prediction_length,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=32)
    # uni2ts 2.0 passes float quantile levels, but gluonts 0.14.3's
    # QuantileForecastGenerator requires string keys ("0.1", ...). Repoint it.
    try:
        from gluonts.model.forecast_generator import QuantileForecastGenerator

        levels = [str(q) for q in model.module.quantile_levels]
        predictor.forecast_generator = QuantileForecastGenerator(levels)
    except Exception:
        pass
    return predictor


# ---------------------------------------------------------------- TabPFN-TS (known covariates only)

def make_tabpfn_predict(pipeline: Any, ids=None) -> PredictFn:
    """Wrap TabPFNTSPipeline.predict_df (item_id schema). Past/static channels
    are dropped by the library itself; only known-future covariates matter."""

    def predict(Y_past: np.ndarray, X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        import pandas as pd

        N, T = Y_past.shape
        H = X_future.shape[1]
        C = X_past.shape[2]
        _ids = ids if ids is not None else [f"s{i}" for i in range(N)]
        base = pd.Timestamp("2020-01-01")
        ctx_parts, fut_parts = [], []
        for i in range(N):
            ctx = pd.DataFrame({"item_id": _ids[i],
                                "timestamp": pd.date_range(base, periods=T, freq="D"),
                                "target": _as_float(Y_past[i])})
            fut = pd.DataFrame({"item_id": _ids[i],
                                "timestamp": pd.date_range(base + pd.Timedelta(days=T), periods=H, freq="D")})
            for c in range(C):
                ctx[f"cov_{c}"] = _as_float(X_past[i, :, c])
                fut[f"cov_{c}"] = _as_float(X_future[i, :, c])
            ctx_parts.append(ctx)
            fut_parts.append(fut)
        out = pipeline.predict_df(context_df=pd.concat(ctx_parts, ignore_index=True),
                                  future_df=pd.concat(fut_parts, ignore_index=True))
        # point forecast lives in the "target" column, indexed by (item_id, timestamp)
        out = out.reset_index()
        vals = []
        for sid in _ids:
            g = out[out["item_id"] == sid].sort_values("timestamp")["target"].to_numpy(dtype=float)
            vals.append(g[:H])
        return np.stack(vals)

    predict.__name__ = "tabpfn_ts"
    return predict


def load_tabpfn():
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    return TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)
