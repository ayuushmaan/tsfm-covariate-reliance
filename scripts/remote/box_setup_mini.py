"""Mini-bench setup cell (mini_bench + mini_cov_tasks globals)."""
import marimo._code_mode as cm

SETUP_CODE = '''from tsfm_audit.adapters import FEV_MINI_YAML as _YML
from fev import Benchmark as _BM
mini_bench = _BM.from_yaml(_YML)
mini_cov_tasks = [
    t for t in mini_bench.tasks
    if getattr(t, "known_dynamic_columns", []) or getattr(t, "past_dynamic_columns", []) or getattr(t, "static_columns", [])
]
print(f"mini tasks: {len(mini_bench.tasks)}, with covariates: {len(mini_cov_tasks)}")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(SETUP_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
