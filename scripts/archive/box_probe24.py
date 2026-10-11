"""Read gluonts/torch/model/predictor.py header (imports + registration context)."""
import pathlib

src = pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages/gluonts/torch/model/predictor.py").read_text().splitlines()
print("\n".join(str(i+1) + ": " + l for i, l in enumerate(src[:70])))
