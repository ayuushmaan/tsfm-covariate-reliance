"""predict_to_numpy registry vs Moirai2Forecast MRO."""
import pathlib

src = pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages/gluonts/model/forecast_generator.py").read_text().splitlines()
print("\n".join(str(i+1) + ": " + l for i, l in enumerate(src[40:100])))
from uni2ts.model.moirai2 import Moirai2Forecast
print("MRO:", [c.__name__ for c in Moirai2Forecast.__mro__][:8])
import torch
print("is nn.Module:", issubclass(Moirai2Forecast, torch.nn.Module))
