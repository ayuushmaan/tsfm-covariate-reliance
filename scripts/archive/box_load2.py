"""Load toto + timesfm25 (fixed), print 2.5 covariate signature for adapter check."""
import marimo._code_mode as cm

LOAD2_CODE = '''import tsfm_audit.adapters_extra as _ax2
import importlib as _ilx2
_ilx2.reload(_ax2)
import inspect as _ix
print("fwc sig:", _ix.signature(_ax2.load_timesfm25.__wrapped__ if hasattr(_ax2.load_timesfm25, "__wrapped__") else _ax2.make_timesfm25_predict))
try:
    loaded["toto"] = _ax2.load_toto()
    print("toto: LOADED", flush=True)
except Exception as e:
    loaded["toto"] = None
    print(f"toto: FAIL {type(e).__name__}: {str(e)[:220]}", flush=True)
try:
    loaded["timesfm25"] = _ax2.load_timesfm25()
    print("timesfm25: LOADED", flush=True)
    print("fwc method sig:", _ix.signature(loaded["timesfm25"].forecast_with_covariates))
except Exception as e:
    loaded["timesfm25"] = None
    print(f"timesfm25: FAIL {type(e).__name__}: {str(e)[:220]}", flush=True)
print("done:", {k: (v is not None) for k, v in loaded.items()})
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(LOAD2_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
