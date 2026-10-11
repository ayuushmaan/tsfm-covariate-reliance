"""uni2ts 2.0 layout: model files + Moirai2Forecast wiring."""
import importlib.metadata as _md
import pathlib

print("requires:", _md.requires("uni2ts"))
root = pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages/uni2ts")
print("top:", sorted(p.name for p in root.iterdir()))
import uni2ts.model.moirai2 as _m2
print("module file:", _m2.__file__)
import inspect
from uni2ts.model.moirai2 import Moirai2Forecast
print(inspect.getsource(Moirai2Forecast.create_predictor))
