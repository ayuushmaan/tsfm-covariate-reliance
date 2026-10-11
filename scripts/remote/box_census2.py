"""Census: per-task series/windows/targets + disk space."""
import marimo._code_mode as cm

CENSUS_CODE = '''from fev import Benchmark as _Benchmark3
_full3 = _Benchmark3.from_yaml("https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml")
import json as _jc2
import shutil as _sh2
_ncov = _jc2.load(open("/marimo/full_cov_tasks.json"))
for _n, _h, _k, _p in _ncov:
    try:
        _t = [t for t in _full3.tasks if t.task_name == _n][0]
        _nt = len(getattr(_t, "target_columns", ["x"]))
        _w0 = _t.get_window(0)
        _pd, _fd = _w0.get_input_data()
        print(f"{_n}: n_series={len(_pd)} nwin={_t.num_windows} ntarget={_nt}", flush=True)
    except Exception as e:
        print(f"{_n}: CENSUS FAIL {type(e).__name__}: {str(e)[:140]}", flush=True)
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(CENSUS_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
