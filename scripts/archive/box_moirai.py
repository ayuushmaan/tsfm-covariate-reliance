"""Purge stale gluonts 0.17 from sys.modules, reimport (expect 0.14.3), retest Moirai."""
import marimo._code_mode as cm

MOIRAI_CODE = '''import sys as _sysg
for _m in [m for m in _sysg.modules if m == "gluonts" or m.startswith("gluonts.")]:
    del _sysg.modules[_m]
import gluonts as _g14
print("gluonts now:", _g14.__version__, flush=True)
import tsfm_audit.adapters_extra as _axg
import importlib as _ilxg
_ilxg.reload(_axg)
loaded["moirai2"] = _axg.load_moirai2(prediction_length=168, context_length=512)
print("predictor recreated", flush=True)
from tsfm_audit.planted import PlantedConfig as _PCg, generate_planted_panel as _gppg
_pg = _gppg(1.0, _PCg(n_series=2, seed=0))
_prg = _axg.make_moirai2_predict(loaded["moirai2"])(_pg["Y_past"][:, :64], _pg["X_past"][:, :64, :], _pg["X_future"])
import numpy as _npg
print("moirai predict:", _prg.shape, "nan:", _npg.isnan(_prg).sum(), flush=True)
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(MOIRAI_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
