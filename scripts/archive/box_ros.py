"""Rossmann window-0 dtype census (read-only, no modeling)."""
import marimo._code_mode as cm

DTYPES_CODE = '''_ros = [t for t in mini_cov_tasks if t.task_name == "rossmann_1W"][0]
_ros_w = _ros.get_window(0)
_ros_p, _ros_f = _ros_w.get_input_data()
print("n_series:", len(_ros_p), "H:", _ros.horizon)
print("past cols:", _ros_p.column_names)
print("fut cols:", _ros_f.column_names)
_r0 = _ros_p[0]
import numpy as _npd
for _k in _ros_p.column_names:
    _v = _r0[_k]
    _a = _npd.asarray(_v)
    if _a.dtype.kind in ("U", "S", "O"):
        print(f"{_k}: dtype={_a.dtype} nunique={len(set(_a.tolist()))} vals={sorted(set(str(x) for x in _a.tolist()))[:8]}")
    else:
        print(f"{_k}: dtype={_a.dtype} shape={_a.shape}")
print("static decl:", _ros.static_columns)
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(DTYPES_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
