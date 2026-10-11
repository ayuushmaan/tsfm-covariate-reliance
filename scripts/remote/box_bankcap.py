"""Bank the capture table as numbers_capture.json for the paper."""
import json as _jd3
import glob as _gd3
from collections import defaultdict as _dd3
import numpy as _npd3

_ceil = _jd3.load(open("/marimo/ceiling.json"))
_rows = []
for _fp in _gd3.glob("/marimo/sweep_*.jsonl"):
    for _ln in open(_fp):
        try:
            _r = _jd3.loads(_ln)
            if "mase_intact" in _r:
                _rows.append(_r)
        except Exception:
            pass
_grp = _dd3(list)
for _r in _rows:
    _grp[(_r["task"], _r["model"])].append(_r)
_out = {}
for (_task, _model), _rs in sorted(_grp.items()):
    _intact = float(_npd3.mean([r["mase_intact"] for r in _rs]))
    _gap = float(_npd3.mean([r["swap_gap"] for r in _rs]))
    _seeds = sorted(r["seed"] for r in _rs)
    _cm = _ceil.get(_task)
    _bl = _grp.get((_task, "bolt"), [])
    _bolt = float(_npd3.mean([r["mase_intact"] for r in _bl])) if _bl else None
    _cap = None
    if _cm is not None and _bolt is not None and _bolt > _cm:
        _cap = (_bolt - _intact) / (_bolt - _cm)
    _out.setdefault(_task, {})[_model] = {"intact": round(_intact, 4), "gap": round(_gap, 4),
                                          "seeds": _seeds, "capture": None if _cap is None else round(_cap, 2)}
_out["_ceiling"] = {k: round(v, 4) for k, v in _ceil.items()}
open("/marimo/numbers_capture.json", "w").write(_jd3.dumps(_out, indent=1))
print("banked numbers_capture.json:", len(_out), "tasks")
