"""Full-batch predict NaN census for both models, all 3 conditions."""
import marimo._code_mode as cm

CENSUS_CODE = '''import numpy as _np5
import tsfm_audit.adapters as _a5
import tsfm_audit.fev_convert as _fc5
importlib.reload(_a5)
importlib.reload(_fc5)
from tsfm_audit.adapters import make_bolt_predict as _mk_b5, make_chronos2_predict as _mk_c5
from tsfm_audit.fev_convert import window_to_arrays as _w5
from tsfm_audit.probes import permute_time as _pt5, swap_across_series as _sw5

_rok5 = [t for t in mini_cov_tasks if t.task_name == "rohlik_orders_1D"][0]
_f5 = _w5(_rok5.get_window(0), max_context=512)
print("fresh shapes:", _f5["Y_past"].shape, _f5["X_past"].shape, _f5["X_future"].shape,
      "nan Xf past-cols:", _np5.isnan(_f5["X_future"][:, :, 4:]).sum(), "of", _f5["X_future"][:, :, 4:].size)
_c5 = _mk_c5(chronos2_pipe, ids=_f5["ids"], past_ts=_f5["past_ts"], fut_ts=_f5["fut_ts"])
_b5 = _mk_b5(bolt_pipe, ids=_f5["ids"], past_ts=_f5["past_ts"], fut_ts=_f5["fut_ts"])
_rng5 = _np5.random.default_rng(0)
_Xfull5 = _np5.concatenate([_f5["X_past"], _f5["X_future"]], axis=1)
_Xsw5 = _sw5(_Xfull5, _rng5)
_Xpm5 = _pt5(_Xfull5, _rng5)
_T5, _H5 = _f5["Y_past"].shape[1], _f5["Y_future"].shape[1]
for _name, _Xf in [("intact", _Xfull5), ("swapped", _Xsw5), ("timeperm", _Xpm5)]:
    _Xp, _Xf2 = _Xf[:, :_T5, :], _Xf[:, _T5:_T5 + _H5, :]
    for _mname, _pfn in [("c2", _c5), ("bolt", _b5)]:
        try:
            _p = _pfn(_f5["Y_past"], _Xp, _Xf2)
            print(f"{_mname}/{_name}: shape={_p.shape} nan={_np5.isnan(_p).sum()} mean={_np5.nanmean(_p):.1f}", flush=True)
        except Exception as _e:
            print(f"{_mname}/{_name}: ERROR {type(_e).__name__}: {str(_e)[:200]}", flush=True)
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(CENSUS_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
