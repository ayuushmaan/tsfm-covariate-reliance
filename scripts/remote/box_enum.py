"""Enumerate full fev-bench covariate tasks: univariate vs multivariate, sizes."""
import marimo._code_mode as cm

ENUM_CODE = '''from fev import Benchmark as _Benchmark2
_full = _Benchmark2.from_yaml("https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml")
print("total tasks:", len(_full.tasks))
_cov = [t for t in _full.tasks if getattr(t, "known_dynamic_columns", []) or getattr(t, "past_dynamic_columns", []) or getattr(t, "static_columns", [])]
print("with covariates:", len(_cov))
import json as _je
_ncov = [(t.task_name, t.horizon, list(getattr(t, "known_dynamic_columns", [])), list(getattr(t, "past_dynamic_columns", []))) for t in _cov]
open("/marimo/full_cov_tasks.json", "w").write(_je.dumps(_ncov, indent=1))
for _n, _h, _k, _p in _ncov:
    print(f"{_n} H={_h} known={len(_k)} past={len(_p)}")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(ENUM_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
