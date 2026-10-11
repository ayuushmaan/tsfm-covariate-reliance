"""Load all remaining checkpoints (sequential, per-model try/except)."""
import marimo._code_mode as cm

LOADALL_CODE = '''import pandas.core.internals as _pi
if not hasattr(_pi, "ArrayManager"):
    _pi.ArrayManager = type("ArrayManager", (), {})
import tsfm_audit.adapters_extra as _ax
import importlib as _ilx
_ilx.reload(_ax)

loaded = {}
def _try(name, fn):
    try:
        loaded[name] = fn()
        print(f"{name}: LOADED", flush=True)
    except Exception as e:
        loaded[name] = None
        print(f"{name}: FAIL {type(e).__name__}: {str(e)[:220]}", flush=True)

_try("tirex", _ax.load_tirex)
_try("t0", _ax.load_t0)
_try("toto", _ax.load_toto)
_try("timesfm3", _ax.load_timesfm3)
_try("timesfm25", _ax.load_timesfm25)
_try("c2synth", lambda: __import__("chronos").Chronos2Pipeline.from_pretrained("autogluon/chronos-2-synth", device_map="cuda"))
def _moirai():
    import tsfm_audit.adapters_extra as _m
    return _m.load_moirai2(prediction_length=168, context_length=512)
_try("moirai2", _moirai)
def _tabpfn():
    import tsfm_audit.adapters_extra as _t
    return _t.load_tabpfn()
_try("tabpfn", _tabpfn)
print("done:", {k: (v is not None) for k, v in loaded.items()})
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(LOADALL_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
