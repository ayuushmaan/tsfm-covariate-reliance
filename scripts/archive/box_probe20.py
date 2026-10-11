"""Full traceback for H=24 Moirai failure."""
import traceback

import numpy as _np
import pandas as _pd
from gluonts.dataset.common import ListDataset

import tsfm_audit.adapters_extra as _axe

try:
    _mp = _axe.load_moirai2(prediction_length=24, context_length=512)
    _ds = ListDataset([{"start": _pd.Timestamp("2020-01-01"), "target": _np.sin(_np.linspace(0, 20, 600))}], freq="D")
    _f = list(_mp.predict(_ds))
    print("OK")
except Exception:
    traceback.print_exc(limit=30)
