"""Check ArrayManager usage in ts_dataframe.py + try a pre-import shim."""
import pathlib

p = pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages/tabpfn_time_series/ts_dataframe.py")
src = p.read_text()
for i, line in enumerate(src.splitlines(), 1):
    if "ArrayManager" in line or "BlockManager" in line:
        print(i, line.strip())

import pandas.core.internals as _pi

class _DummyAM:
    pass

_pi.ArrayManager = _DummyAM
import importlib

try:
    importlib.import_module("tabpfn_time_series")
    print("tabpfn: OK via shim")
except Exception as e:
    print("shim FAILED:", str(e)[:200])
