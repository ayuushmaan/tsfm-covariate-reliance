"""Locate predict_to_numpy + Moirai2Module.forward source."""
import subprocess

print(subprocess.run(["grep", "-rn", "def predict_to_numpy", "/tmp/uv-venv/lib/python3.13/site-packages/gluonts/"],
                     capture_output=True, text=True).stdout[:500])
print(subprocess.run(["grep", "-n", "predict_to_numpy", "/tmp/uv-venv/lib/python3.13/site-packages/gluonts/model/forecast_generator.py"],
                     capture_output=True, text=True).stdout[:300])
import inspect
from uni2ts.model.moirai2 import Moirai2Module
print(inspect.getsource(Moirai2Module.forward))
