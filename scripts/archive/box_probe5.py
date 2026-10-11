"""Toto _from_pretrained source + moirai full traceback + tabpfn full license text."""
import inspect
import traceback

from toto.model.toto import Toto

try:
    print(inspect.getsource(Toto._from_pretrained))
except Exception as e:
    print("no source:", e)

print("===== moirai traceback =====")
import tsfm_audit.adapters_extra as _axe
import tsfm_audit.harness as _hx
from tsfm_audit.planted import PlantedConfig as _PCm, generate_planted_panel as _gppm

try:
    _px = _gppm(1.0, _PCm(n_series=2, seed=0))
    _kw = dict(Y_past=_px["Y_past"], X_past=_px["X_past"], X_future=_px["X_future"], Y_future=_px["Y_future"])
    _pred = _axe.make_moirai2_predict(loaded["moirai2"])
    _pred(_px["Y_past"][:, :64], _px["X_past"][:, :64, :], _px["X_future"])
    print("moirai mini predict OK")
except Exception:
    traceback.print_exc(limit=25)

print("===== tabpfn license text =====")
try:
    _tp = _axe.make_tabpfn_predict(loaded["tabpfn"])
    _tp(_px["Y_past"][:, :64], _px["X_past"][:, :64, :], _px["X_future"])
    print("tabpfn mini predict OK")
except Exception:
    traceback.print_exc(limit=12)
