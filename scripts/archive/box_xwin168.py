"""Cross-window probe: uci + proenfo (H=168 long horizon)."""
import marimo._code_mode as cm

WIN168_CODE = '''import tsfm_audit.adapters as _aq
import tsfm_audit.fev_convert as _fq
import tsfm_audit.harness as _hq
import importlib as _ilq
for _mm in (_aq, _fq, _hq):
    _ilq.reload(_mm)
from tsfm_audit.adapters import make_bolt_predict as _mk_bq, make_chronos2_predict as _mk_cq
from tsfm_audit.fev_convert import window_to_arrays as _wq
from tsfm_audit.harness import run_probe_windows as _rpq
import json as _jq

win168_rows = []
for _tname in ["uci_air_quality_1H", "proenfo_gfc14"]:
    _tw = [t for t in mini_cov_tasks if t.task_name == _tname][0]
    _wins = []
    for _wi in range(_tw.num_windows):
        _a = _wq(_tw.get_window(_wi), max_context=512)
        _wins.append({"Y_past": _a["Y_past"][0], "X_past": _a["X_past"][0],
                      "X_future": _a["X_future"][0], "Y_future": _a["Y_future"][0]})
    print(f"{_tname}: {len(_wins)} windows H={_tw.horizon}", flush=True)
    _wc = _mk_cq(chronos2_pipe)
    _wb = _mk_bq(bolt_pipe)
    for _s in [0, 1]:
        for _mn, _mp in [("chronos2", _wc), ("bolt", _wb)]:
            _rr = _rpq(_mn, f"{_tname}/xwin", _wins, predict=_mp, seed=_s).to_dict()
            win168_rows.append(_rr)
            print(f"  seed={_s} {_mn}: intact={_rr['mase_intact']:.4f} swapped={_rr['mase_swapped']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
with open("/marimo/grid_xwin168.jsonl", "w") as _f:
    for _r in win168_rows:
        _f.write(_jq.dumps(_r) + "\\n")
print(f"banked {len(win168_rows)} rows")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(WIN168_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
