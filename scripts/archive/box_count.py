"""Count series per covariate task + window API surface for ground truth."""
import marimo._code_mode as cm

COUNT_CODE = '''for _t in mini_cov_tasks:
    _w = _t.get_window(0)
    _p, _f = _w.get_input_data()
    print(f"{_t.task_name}: n_series={len(_p)} H={_t.horizon}")
print([_m for _m in dir(_w) if not _m.startswith("_")])
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(COUNT_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
