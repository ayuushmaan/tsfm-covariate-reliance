"""Toto scale diagnosis + TimesFM3 univariate check (rohlik)."""
import marimo._code_mode as cm

DIAG2_CODE = '''import tsfm_audit.adapters_extra as _ad2
import importlib as _il2
_il2.reload(_ad2)
import numpy as _np2
from tsfm_audit.metrics import mase as _m2

_ro = [t for t in mini_cov_tasks if t.task_name == "rohlik_orders_1D"][0]
from tsfm_audit.fev_convert import window_to_arrays as _w2
_ra2 = _w2(_ro.get_window(0), max_context=512)
print("channel stds:", _np2.std(_ra2["X_past"].reshape(-1, 13), axis=0).round(3), flush=True)
print("Y std:", _np2.std(_ra2["Y_past"]), "Y mean:", _np2.mean(_ra2["Y_past"]), flush=True)

_tc = _ad2.make_toto_predict(loaded["toto"], num_samples=32)
# (a) standardized panel: does scale break Toto?
_mu, _sd = _ra2["Y_past"].mean(), _ra2["Y_past"].std()
try:
    _ps = _tc((_ra2["Y_past"] - _mu) / _sd, _ra2["X_past"] / (_np2.std(_ra2["X_past"].reshape(-1, 13), axis=0, keepdims=True) + 1e-8), _ra2["X_future"])
    print("toto standardized: OK nan=", _np2.isnan(_ps).sum(), flush=True)
except Exception as e:
    print("toto standardized FAIL:", str(e)[:160], flush=True)
# (b) drop zero-variance past channels only
_var = _np2.std(_ra2["X_past"], axis=1).mean(axis=0)
_keep = [c for c in range(13) if _var[c] > 1e-9]
print("keep channels:", _keep, flush=True)
try:
    _pk = _tc(_ra2["Y_past"], _ra2["X_past"][:, :, _keep], _ra2["X_future"][:, :, _keep])
    print("toto drop-const: OK nan=", _np2.isnan(_pk).sum(), flush=True)
except Exception as e:
    print("toto drop-const FAIL:", str(e)[:160], flush=True)

# (c) timesfm3 univariate vs covariate on rohlik
import tsfm_audit.adapters_extra as _ax3
_fc3 = loaded["timesfm3"]
_ctxs = [(_ra2["Y_past"][i]).astype("float32")[None, :] for i in range(7)]
_o_nc = list(_fc3.predict_batch(contexts=_ctxs, horizon=61, past_only_covariates=[None]*7, past_future_covariates=[None]*7, return_quantiles=False))
import numpy as _np3
_pnc = _np3.stack([_np3.asarray(o.forecast, dtype=float).reshape(-1)[:61] for o in _o_nc])
_mnc = float(_np3.nanmean([_m2(_ra2["Y_future"][i], _pnc[i], _ra2["Y_past"][i]) for i in range(7)]))
print(f"timesfm3 univariate intact={_mnc:.4f} (vs covariate 0.9244)", flush=True)
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(DIAG2_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
