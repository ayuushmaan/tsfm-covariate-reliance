"""Reship is done separately; this edits box cell jBbI in place (reload fev_convert) and runs it."""
import marimo._code_mode as cm

async with cm.get_context() as ctx:
    for c in ctx.cells:
        if c.id == "jBbI":
            print("old:", repr((c.code or "")[:300]))
            new = c.code.replace(
                "from tsfm_audit.fev_convert import window_to_arrays as _w2a2",
                "import tsfm_audit.fev_convert as _fc2\nimportlib.reload(_fc2)\nfrom tsfm_audit.fev_convert import window_to_arrays as _w2a2",
            )
            new = new.replace(
                "import tsfm_audit.adapters as _a2\nimportlib.reload(_a2)",
                "import tsfm_audit.adapters as _a2\nimportlib.reload(_a2)",
            )
            assert new != c.code, "no change?!"
            ctx.edit_cell("jBbI", new)
            ctx.run_cell("jBbI")
            print("edited+run")
            break
print("queued")
