"""Fresh capture table (verify data path end to end)."""
import json as _jd2
import glob as _gd2
from collections import defaultdict as _dd2
import numpy as _npd2

_ceil = _jd2.load(open("/marimo/ceiling.json"))
_rows = []
for _fp in _gd2.glob("/marimo/sweep_*.jsonl"):
    for _ln in open(_fp):
        try:
            _r = _jd2.loads(_ln)
            if "mase_intact" in _r:
                _rows.append(_r)
        except Exception:
            pass
_grp = _dd2(list)
for _r in _rows:
    _grp[(_r["task"], _r["model"])].append(_r)
print(f"{'task':38s} {'model':10s} {'intact':>7s} {'gap':>7s} {'capture':>7s}")
for (_task, _model), _rs in sorted(_grp.items()):
    _intact = float(_npd2.mean([r["mase_intact"] for r in _rs]))
    _gap = float(_npd2.mean([r["swap_gap"] for r in _rs]))
    _cm = _ceil.get(_task, None)
    _bl = _grp.get((_task, "bolt"), [])
    _bolt = float(_npd2.mean([r["mase_intact"] for r in _bl])) if _bl else float("nan")
    _cap = "n/a" if (_cm is None or not (_bolt > _cm)) else f"{(_bolt - _intact) / (_bolt - _cm):+.2f}"
    print(f"{_task:38s} {_model:10s} {_intact:7.3f} {_gap:+7.3f} {_cap:>7s}")
