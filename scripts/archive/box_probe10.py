"""Locate all gluonts copies; remove uv-venv shadow; verify 0.14.3."""
import pathlib
import shutil
import subprocess
import sys

for d in ("/tmp/uv-venv/lib/python3.13/site-packages", "/usr/local/lib/python3.13/site-packages"):
    hits = sorted(pathlib.Path(d).glob("gluonts*"))
    print(d, "->", [h.name for h in hits])

shutil.rmtree("/tmp/uv-venv/lib/python3.13/site-packages/gluonts", ignore_errors=True)
for pat in pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages").glob("gluonts-*.dist-info"):
    shutil.rmtree(pat, ignore_errors=True)
    print("removed dist-info:", pat.name)

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "gluonts==0.14.3"],
                   capture_output=True, text=True, timeout=600)
print("install rc:", r.returncode, (r.stderr or "")[-300:])
import gluonts
print("now:", gluonts.__file__, gluonts.__version__)
