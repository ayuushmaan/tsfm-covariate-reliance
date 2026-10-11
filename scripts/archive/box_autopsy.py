"""Numeric MASE autopsy for series 1."""
import marimo._code_mode as cm

AUTOPSY_CODE = '''import numpy as _np7
import tsfm_audit.adapters as _a7
importlib.reload(_a7)
from tsfm_audit.adapters import make_bolt_predict as _mk_b7
_b7 = _mk_b7(bolt_pipe, ids=fev_arrays["ids"], past_ts=fev_arrays["past_ts"], fut_ts=fev_arrays["fut_ts"])
_pb7 = _b7(fev_arrays["Y_past"], fev_arrays["X_past"], fev_arrays["X_future"])
_yt = fev_arrays["Y_future"][1].astype(float)
_yp = _pb7[1].astype(float)
_ytr = fev_arrays["Y_past"][1].astype(float)
print("truth: nan=", _np7.isnan(_yt).sum(), "inf=", _np7.isinf(_yt).sum(), "neg=", (_yt < 0).sum())
print("pred: nan=", _np7.isnan(_yp).sum(), "inf=", _np7.isinf(_yp).sum())
print("train: nan=", _np7.isnan(_ytr).sum(), "inf=", _np7.isinf(_ytr).sum(), "len=", len(_ytr))
_valid = ~_np7.isnan(_yt) & ~_np7.isnan(_yp)
print("valid:", _valid.sum())
print("err mean:", _np7.mean(_np7.abs(_yt[_valid] - _yp[_valid])))
_ss = _ytr[~_np7.isnan(_ytr)]
print("scale:", _np7.mean(_np7.abs(_ss[1:] - _ss[:-1])))
print("truth[55:61]:", _yt[55:61])
print("pred[55:61]:", _yp[55:61])
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(AUTOPSY_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
