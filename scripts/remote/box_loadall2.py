"""Load all 10 checkpoints on a fresh box (uses fixed repo loaders)."""
import marimo._code_mode as cm

LOADALL2 = '''import tsfm_audit.adapters as _ada
import tsfm_audit.adapters_extra as _axa
import importlib as _ilx
_ilx.reload(_ada)
_ilx.reload(_axa)

loaded = {}
def _try(_name, _fn):
    try:
        loaded[_name] = _fn()
        print(f"{_name}: LOADED", flush=True)
    except Exception as e:
        loaded[_name] = None
        print(f"{_name}: FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)

_try("chronos2", _ada.load_chronos2)
_try("bolt", _ada.load_bolt)
_try("tirex", _axa.load_tirex)
_try("t0", _axa.load_t0)
_try("toto", _axa.load_toto)
_try("timesfm3", _axa.load_timesfm3)
_try("timesfm25", _axa.load_timesfm25)
_try("c2synth", lambda: __import__("chronos").Chronos2Pipeline.from_pretrained("autogluon/chronos-2-synth", device_map="cuda"))
_try("moirai2", lambda: _axa.load_moirai2(prediction_length=168, context_length=512))
_try("tabpfn", _axa.load_tabpfn)
chronos2_pipe = loaded["chronos2"]
bolt_pipe = loaded["bolt"]
print("done:", {k: (v is not None) for k, v in loaded.items()})
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(LOADALL2, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
