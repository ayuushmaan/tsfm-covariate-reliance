"""Build scripts/remote/ship_package.py: base64 bundle of src/tsfm_audit for the GPU box."""
import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src" / "tsfm_audit"
FILES = ["__init__.py", "probes.py", "metrics.py", "planted.py", "ceiling.py", "harness.py", "adapters.py", "adapters_extra.py", "fev_convert.py"]

blobs = {f: base64.b64encode((SRC / f).read_bytes()).decode() for f in FILES}
lines = [
    "import base64, pathlib",
    "d = pathlib.Path('/marimo/tsfm_audit'); d.mkdir(exist_ok=True)",
    "blobs = {",
]
lines += [f"'{f}': '{b}'," for f, b in blobs.items()]
lines += [
    "}",
    "[(d / k).write_bytes(base64.b64decode(v)) for k, v in blobs.items()]",
    "import tsfm_audit as t; print('shipped', t.__version__)",
]
out = pathlib.Path(__file__).resolve().parent / "ship_package.py"
out.write_text("\n".join(lines) + "\n")
print("wrote", out, out.stat().st_size, "bytes")
