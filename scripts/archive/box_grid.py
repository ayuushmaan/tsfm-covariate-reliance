"""Grid: rohlik + uk_covid x seeds 0,1,2 x both models. Banks to /marimo/grid_mini.jsonl."""
import marimo._code_mode as cm

GRID2_CODE = '''import tsfm_audit.adapters as _ag
import tsfm_audit.fev_convert as _fg
import tsfm_audit.harness as _hg
import tsfm_audit.metrics as _mg
import importlib as _ilg
for _mm in (_ag, _fg, _hg, _mg):
    _ilg.reload(_mm)
from tsfm_audit.adapters import make_bolt_predict as _mk_bg, make_chronos2_predict as _mk_cg
from tsfm_audit.fev_convert import window_to_arrays as _wg
from tsfm_audit.harness import run_probe as _rpg
import json as _jg

grid_rows = []
for _tname in ["rohlik_orders_1D", "uk_covid_nation_1D/cumulative"]:
    _gt = [t for t in mini_cov_tasks if t.task_name == _tname][0]
    _ga = _wg(_gt.get_window(0), max_context=512)
    print(f"{_tname}: N={_ga['Y_past'].shape[0]} T={_ga['Y_past'].shape[1]} C={_ga['X_past'].shape[2]} known={_ga['n_known']} H={_ga['Y_future'].shape[1]}", flush=True)
    _gc = _mk_cg(chronos2_pipe, ids=_ga["ids"], past_ts=_ga["past_ts"], fut_ts=_ga["fut_ts"])
    _gb = _mk_bg(bolt_pipe, ids=_ga["ids"], past_ts=_ga["past_ts"], fut_ts=_ga["fut_ts"])
    _gkw = dict(Y_past=_ga["Y_past"], X_past=_ga["X_past"], X_future=_ga["X_future"], Y_future=_ga["Y_future"])
    for _s in [0, 1, 2]:
        for _mn, _mp in [("chronos2", _gc), ("bolt", _gb)]:
            _rr = _rpg(_mn, f"{_tname}/w0", **_gkw, predict=_mp, seed=_s).to_dict()
            grid_rows.append(_rr)
            print(f"  seed={_s} {_mn}: intact={_rr['mase_intact']:.4f} swapped={_rr['mase_swapped']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
with open("/marimo/grid_mini.jsonl", "w") as _f:
    for _r in grid_rows:
        _f.write(_jg.dumps(_r) + "\\n")
print(f"banked {len(grid_rows)} rows")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(GRID2_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
