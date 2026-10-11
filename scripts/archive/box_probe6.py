"""Toto checkpoint config keys + uni2ts gluonts pin."""
import importlib.metadata as _md
import inspect

print("uni2ts requires:")
for r in (_md.requires("uni2ts") or []):
    if "gluonts" in r.lower() or "torch" in r.lower() or "numpy" in r.lower():
        print(" ", r)

from toto.model.toto import Toto
print("load_from_checkpoint sig:", inspect.signature(Toto.load_from_checkpoint))

import json
import urllib.request
cfg = json.load(urllib.request.urlopen("https://huggingface.co/Datadog/Toto-Open-Base-1.0/raw/main/config.json", timeout=60))
print("config keys:", list(cfg.keys()))
print("d_model/params:", {k: cfg.get(k) for k in ("d_model", "num_layers", "patch_size") if k in cfg})
