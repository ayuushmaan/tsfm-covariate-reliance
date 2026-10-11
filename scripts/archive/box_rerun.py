"""Add harness+metrics reloads to jBbI and re-run (kills stale-module NaNs)."""
import marimo._code_mode as cm

async with cm.get_context() as ctx:
    for c in ctx.cells:
        if c.id == "jBbI":
            code = c.code or ""
            anchor = "import tsfm_audit.adapters as _a2\nimportlib.reload(_a2)"
            fresh = (
                "import tsfm_audit.adapters as _a2\nimportlib.reload(_a2)\n"
                "import tsfm_audit.harness as _h2\nimportlib.reload(_h2)\n"
                "import tsfm_audit.metrics as _m2\nimportlib.reload(_m2)"
            )
            if anchor in code:
                ctx.edit_cell("jBbI", code.replace(anchor, fresh))
                print("reloaded all")
            else:
                print("anchor missing; head:", repr(code[:200]))
            ctx.run_cell("jBbI")
            break
print("queued")
