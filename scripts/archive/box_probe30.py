"""Read 2.5 torch compile + ForecastConfig fields."""
import pathlib

p = pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages/timesfm/timesfm_2p5/timesfm_2p5_torch.py").read_text().splitlines()
print("\n".join(str(i+1) + ": " + l for i, l in enumerate(p[370:410])))
c = pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages/timesfm/configs.py").read_text().splitlines()
print("===== configs =====")
print("\n".join(str(i+1) + ": " + l for i, l in enumerate(c[:80])))
