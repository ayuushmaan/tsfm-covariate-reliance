"""TimesFM 2.5: package layout + ForecastConfig + compile usage."""
import pathlib
import subprocess

tfm = pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages/timesfm")
print("2p5 dir:", sorted(p.name for p in (tfm / "timesfm_2p5").iterdir()))
print(subprocess.run(["grep", "-rn", "def compile\|class ForecastConfig\|from_pretrained",
                      str(tfm / "timesfm_2p5")], capture_output=True, text=True).stdout[:1200])
print(subprocess.run(["grep", "-rn", "compile(",
                      "/tmp/uv-venv/lib/python3.13/site-packages/timesfm/__init__.py"],
                     capture_output=True, text=True).stdout[:400])
