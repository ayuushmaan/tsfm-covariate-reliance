"""Confirm rohlik intact number with fully fresh conversion + wrappers."""
import marimo._code_mode as cm

CONFIRM_CODE = '''import tsfm_audit.adapters as _ak
import tsfm_audit.fev_convert as _fk
import importlib as _ilk
for _mm in (_ak, _fk):
    _ilk.reload(_mm)
from tsfm_audit.adapters import make_chronos2_predict as _mk_ck
from tsfm_audit.fev_convert import window_to_arrays as _wk
from tsfm_audit.metrics import mase as _mk
import numpy as _npk

_rok = [t for t in mini_cov_tasks if t.task_name == "rohlik_orders_1D"][0]
_fk2 = _wk(_rok.get_window(0), max_context=512)
_ck = _mk_ck(chronos2_pipe, ids=_fk2["ids"], past_ts=_fk2["past_ts"], fut_ts=_fk2["fut_ts"])
_pk = _ck(_fk2["Y_past"], _fk2["X_past"], _fk2["X_future"])
print("pred mean:", _npk.nanmean(_pk), "nan:", _npk.isnan(_pk).sum())
print("per-series mase:", [_mk(_fk2["Y_future"][i], _pk[i], _fk2["Y_past"][i]) for i in range(7)])
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(CONFIRM_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
