"""Bypass inspect: uni2ts transform batch keys + raw module output shape."""
import numpy as _np
import torch

from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

_module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small").eval()
_model = Moirai2Forecast(module=_module, prediction_length=24, context_length=64,
                         target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
import pandas as _pd
from gluonts.dataset.common import ListDataset
from gluonts.transform import TestSplitSampler
from gluonts.transform.split import TFTInstanceSplitter

ds = ListDataset([{"start": _pd.Timestamp("2020-01-01"), "target": _np.sin(_np.linspace(0, 20, 200))}], freq="D")
tf = _model.get_default_transform() + TFTInstanceSplitter(
    instance_sampler=TestSplitSampler(), past_length=_model.past_length,
    future_length=24, observed_value_field="observed_target")
entries = list(tf.iter(ds))
print("entry keys:", sorted(entries[0].keys()))
batch = {k: (torch.as_tensor(v).unsqueeze(0) if not isinstance(v, torch.Tensor) else v.unsqueeze(0))
         for k, v in entries[0].items() if k != "forecast_start"}
print("batch shapes:", {k: tuple(v.shape) for k, v in batch.items()})
print("forecast_start:", entries[0]["forecast_start"])
with torch.no_grad():
    out = _module(training_mode=False, **{k: v for k, v in batch.items() if k in (
        "target", "observed_mask", "sample_id", "time_id", "variate_id", "prediction_mask")})
print("raw out:", type(out), getattr(out, "shape", None))
print("levels:", _module.quantile_levels)
