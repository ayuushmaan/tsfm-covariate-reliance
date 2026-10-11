"""Debug capture comparison for rohlik/epf."""
import json as _jd

_ceil = _jd.load(open("/marimo/ceiling.json"))
print("keys:", list(_ceil.keys()))
for _t in ["rohlik_orders_1D/w0", "epf_np/xwin", "uk_covid_nation_1D/cumulative/w0"]:
    _v = _ceil.get(_t)
    print(repr(_t), type(_v), _v)
import glob as _gd
_n = 0
for _fp in _gd.glob("/marimo/sweep_A1.jsonl"):
    for _ln in open(_fp):
        _r = _jd.loads(_ln)
        if _r["model"] == "bolt":
            print("bolt row task:", repr(_r["task"]), _r["mase_intact"])
            _n += 1
            if _n > 2:
                break
