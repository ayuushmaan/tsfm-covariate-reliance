"""t0 predict signature + timesfm2.5 ForecastConfig usage."""
import inspect
import subprocess

from t0 import T0Forecaster
print("t0 predict sig:", inspect.signature(T0Forecaster.predict))
print(subprocess.run(["grep", "-rn", "ForecastConfig(", "/tmp/uv-venv/lib/python3.13/site-packages/timesfm/",
                      "--include=*.py"], capture_output=True, text=True).stdout[:1200])
print(subprocess.run(["grep", "-rn", "compile(", "/tmp/uv-venv/lib/python3.13/site-packages/timesfm/timesfm_2p5_torch.py"],
                     capture_output=True, text=True).stdout[:600])
