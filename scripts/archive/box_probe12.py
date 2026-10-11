"""Compare gluonts forecast_generator expectations vs Moirai2Module returns."""
import pathlib

g = pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages/gluonts/model/forecast_generator.py").read_text().splitlines()
print("---- gluonts forecast_generator[100:140] ----")
print("\n".join(f"{i+1}: {l}" for i, l in enumerate(g[99:140], start=99)))
import uni2ts
print("uni2ts version:", getattr(uni2ts, "__version__", "?"), uni2ts.__file__)
u = pathlib.Path(uni2ts.__file__).parent / "model" / "moirai2.py"
src = u.read_text()
import re
for m in re.finditer(r"def (forward|create_predictor|__call__)", src):
    print("==== " + m.group(1) + " ====")
    lines = src.splitlines()
    start = src[:m.start()].count(chr(10))
    print("\n".join(str(start+i+1) + ": " + l for i, l in enumerate(lines[start:start+40])))
    print("...")
