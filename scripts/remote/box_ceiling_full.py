"""Full-46 ceilings on Box B (same protocol as sweep; bank ceiling_full.json)."""
import marimo._code_mode as cm

CEILF_CODE = '''import tsfm_audit.ceiling as _clf
import importlib as _ilf
_ilf.reload(_clf)
from tsfm_audit.ceiling import ceiling_ridge as _crf
from tsfm_audit.fev_convert import window_to_arrays as _wcf
from tsfm_audit.metrics import mase as _mcf
import numpy as _npf
import json as _jf
from fev import Benchmark as _B9f
_tasks46 = [t for t in _B9f.from_yaml("https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml").tasks
            if getattr(t, "known_dynamic_columns", []) or getattr(t, "past_dynamic_columns", []) or getattr(t, "static_columns", [])]
print(f"covariate tasks: {len(_tasks46)}", flush=True)

def _mmf(_Yf, _P, _Yp):
    return float(_npf.nanmean([_mcf(_Yf[i], _P[i], _Yp[i]) for i in range(len(_Yp))]))

def _nvf(_Yp, _H):
    return _npf.repeat(_Yp[:, -1:], _H, axis=1)

_ceilf = {}
for _t in _tasks46:
    _w0 = _t.get_window(0)
    _pd, _fd = _w0.get_input_data()
    try:
        if len(_pd) > 1:
            _a = _wcf(_w0, max_context=512)
            _ix = list(range(len(_a["ids"])))
            if len(_ix) > 64:
                _ix = sorted(__import__("random").Random(0).sample(_ix, 64))
            _skip = tuple(_a["col_names"].index(c) for c in _a["cat_maps"])
            _Pr = _crf(_a["Y_past"][_ix], _a["X_past"][_ix], _a["X_future"][_ix], skip_cols=_skip)
            _mr = _mmf(_a["Y_future"][_ix], _Pr, _a["Y_past"][_ix])
            _mn = _mmf(_a["Y_future"][_ix], _nvf(_a["Y_past"][_ix], _a["Y_future"].shape[1]), _a["Y_past"][_ix])
            _key = f"{_t.task_name}/w0"
        else:
            _ms = []
            for _wi in range(_t.num_windows):
                _a = _wcf(_t.get_window(_wi), max_context=512)
                _skip = tuple(_a["col_names"].index(c) for c in _a["cat_maps"])
                _Pr = _crf(_a["Y_past"], _a["X_past"], _a["X_future"], skip_cols=_skip)
                _mr = _mmf(_a["Y_future"], _Pr, _a["Y_past"])
                _mn = _mmf(_a["Y_future"], _nvf(_a["Y_past"], _a["Y_future"].shape[1]), _a["Y_past"])
                _ms.append(min(_mr, _mn))
            _key = f"{_t.task_name}/xwin"
            _mr, _mn = float(_npf.nanmean(_ms)), float(_npf.nanmean(_ms))
        _ceilf[_key] = min(_mr, _mn)
        print(f"{_key}: ridge={_mr:.4f} naive={_mn:.4f}", flush=True)
    except Exception as e:
        print(f"{_t.task_name}: CEIL FAIL {type(e).__name__}: {str(e)[:150]}", flush=True)
with open("/marimo/ceiling_full.json", "w") as _f:
    _f.write(_jf.dumps(_ceilf, indent=1))
print(f"banked ceiling_full.json ({len(_ceilf)} tasks)")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(CEILF_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
