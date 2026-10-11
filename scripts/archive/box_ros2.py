"""Rossmann 32-store GPU validation with categorical path."""
import marimo._code_mode as cm

ROS_CODE = '''import tsfm_audit.adapters as _ar
import tsfm_audit.fev_convert as _fr
import tsfm_audit.harness as _hr
import tsfm_audit.metrics as _mr
import importlib as _ilr
for _mm in (_ar, _fr, _hr, _mr):
    _ilr.reload(_mm)
from tsfm_audit.adapters import make_bolt_predict as _mk_br, make_chronos2_predict as _mk_cr
from tsfm Audit.fev_convert import window_to_arrays as _wr
'''

# NOTE: fixed typo below (tsfm Audit -> tsfm_audit) before sending
ROS_CODE = ROS_CODE.replace("tsfm Audit", "tsfm_audit") + '''from tsfm_audit.harness import run_probe as _rpr
_ros2 = [t for t in mini_cov_tasks if t.task_name == "rossmann_1W"][0]
_ra = _wr(_ros2.get_window(0), max_context=512)
print("full:", _ra["Y_past"].shape, _ra["X_past"].shape, "| cats:", {k: v for k, v in _ra["cat_maps"].items()})
_N = 32
_sub = dict(Y_past=_ra["Y_past"][:_N], X_past=_ra["X_past"][:_N], X_future=_ra["X_future"][:_N], Y_future=_ra["Y_future"][:_N])
_meta = dict(ids=_ra["ids"][:_N], past_ts=_ra["past_ts"][:_N], fut_ts=_ra["fut_ts"][:_N],
             col_names=_ra["col_names"], known_names=_ra["known_names"], cat_maps=_ra["cat_maps"])
_rc = _mk_cr(chronos2_pipe, **_meta)
_rb = _mk_br(bolt_pipe, **_meta)
ros_rows = []
ros_rows.append(_rpr("chronos2", "rossmann_1W/w0x32", **_sub, predict=_rc, seed=0).to_dict())
ros_rows.append(_rpr("bolt", "rossmann_1W/w0x32", **_sub, predict=_rb, seed=0).to_dict())
for _r in ros_rows:
    print(f"{_r['model']}: intact={_r['mase_intact']:.4f} swapped={_r['mase_swapped']:.4f} gap={_r['swap_gap']:+.4f}", flush=True)
import json as _jr
with open("/marimo/smoke_rossmann.jsonl", "w") as _f:
    for _r in ros_rows:
        _f.write(_jr.dumps(_r) + "\\n")
print("banked")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(ROS_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
