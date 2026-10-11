"""Full Moirai traceback + QuantileForecast init source (0.14.3)."""
import inspect
import traceback

from gluonts.model.forecast import QuantileForecast
print(inspect.getsource(QuantileForecast.__init__))
print("===== live traceback =====")
import tsfm_audit.adapters_extra as _axe
from tsfm_audit.planted import PlantedConfig as _PCq, generate_planted_panel as _gppq

try:
    _pq = _gppq(1.0, _PCq(n_series=1, seed=0))
    _prq = _axe.make_moirai2_predict(loaded["moirai2"])(_pq["Y_past"][:, :64], _pq["X_past"][:, :64, :], _pq["X_future"])
    print("OK", _prq.shape)
except Exception:
    traceback.print_exc(limit=40)
