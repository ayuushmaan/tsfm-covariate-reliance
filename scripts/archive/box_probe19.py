"""Moirai with clean horizons: does prediction_length=24 work?"""
import numpy as _np
import pandas as _pd
from gluonts.dataset.common import ListDataset

import tsfm_audit.adapters_extra as _axe

for _H in [24, 61, 168]:
    try:
        _mp = _axe.load_moirai2(prediction_length=_H, context_length=512)
        _ds = ListDataset([{"start": _pd.Timestamp("2020-01-01"), "target": _np.sin(_np.linspace(0, 20, 600))}], freq="D")
        _f = list(_mp.predict(_ds))
        print(_H, "OK mean0:", float(_np.asarray(_f[0].mean)[:3].mean()))
    except Exception as e:
        print(_H, "FAIL:", str(e)[:160])
