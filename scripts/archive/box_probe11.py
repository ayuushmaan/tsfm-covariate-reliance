"""What does pip think gluonts is? Config, artifact, installed version line."""
import pathlib
import subprocess
import sys


def run(args):
    r = subprocess.run([sys.executable, "-m", "pip"] + args,
                       capture_output=True, text=True, timeout=600)
    return r.returncode, ((r.stdout or "") + (r.stderr or ""))[-800:]


rc, out = run(["show", "gluonts"])
print("show rc:", rc)
print(out)
rc, out = run(["config", "list"])
print("config:", out.replace("\n", " | ")[:400])
rc, out = run(["download", "gluonts==0.14.3", "--no-deps", "-d", "/tmp/gdl", "--no-build-isolation"])
print("download rc:", rc, out.replace("\n", " | ")[:500])
print("downloaded:", [p.name for p in pathlib.Path("/tmp/gdl").glob("*")] if pathlib.Path("/tmp/gdl").exists() else None)
init = pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages/gluonts/__init__.py")
print("init head:", init.read_text()[:300] if init.exists() else "MISSING")
