"""Registry keys vs MRO + torch identity."""
import torch

print("torch:", torch.__version__, torch.__file__)
from gluonts.model.forecast_generator import predict_to_numpy
print("registry:", list(predict_to_numpy.registry.keys()))
from uni2ts.model.moirai2 import Moirai2Forecast
import torch.nn as _nn
print("subclass:", issubclass(Moirai2Forecast, _nn.Module))
print("same nn.Module:", _nn.Module is __import__("torch").nn.Module)
try:
    print("dispatch:", predict_to_numpy.dispatch(Moirai2Forecast))
except Exception as e:
    print("dispatch FAIL:", e)
import lightning.pytorch as _lp
print("lightning torch:", _lp.__file__)
