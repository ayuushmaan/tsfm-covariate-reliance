"""Create+run the checkpoint-loading cell in the live notebook (via cm)."""
import marimo._code_mode as cm

LOAD_CODE = '''from tsfm_audit.adapters import load_bolt, load_chronos2, make_bolt_predict, make_chronos2_predict

chronos2_pipe = load_chronos2("cuda")
bolt_pipe = load_bolt("cuda")
chronos2_predict = make_chronos2_predict(chronos2_pipe)
bolt_predict = make_bolt_predict(bolt_pipe)
print("checkpoints loaded")
'''

GRID_CODE = '''from tsfm_audit.harness import run_probe
from tsfm_audit.planted import PlantedConfig, generate_planted_panel

smoke_rows = []
smoke_betas = [0.0, 0.5, 1.0]
for _beta in smoke_betas:
    _panel = generate_planted_panel(_beta, PlantedConfig(n_series=8, seed=0))
    _kw = dict(Y_past=_panel["Y_past"], X_past=_panel["X_past"], X_future=_panel["X_future"], Y_future=_panel["Y_future"])
    smoke_rows.append(run_probe("chronos2", f"planted_beta{_beta:g}", **_kw, predict=chronos2_predict, seed=0).to_dict())
    smoke_rows.append(run_probe("bolt", f"planted_beta{_beta:g}", **_kw, predict=bolt_predict, seed=0).to_dict())
    print(f"beta={_beta:.1f} chronos2_gap={smoke_rows[-2]['swap_gap']:+.4f} bolt_gap={smoke_rows[-1]['swap_gap']:+.4f}", flush=True)
'''

PLOT_CODE = '''import matplotlib.pyplot as plt

_fig, _ax = plt.subplots(figsize=(6, 3.5))
_c = [r["swap_gap"] for r in smoke_rows if r["model"] == "chronos2"]
_b = [r["swap_gap"] for r in smoke_rows if r["model"] == "bolt"]
_ax.plot(smoke_betas, _c, "s-", label="chronos2 (expect rising)")
_ax.plot(smoke_betas, _b, "o--", label="bolt control (expect ~0)")
_ax.set_xlabel("planted effect beta")
_ax.set_ylabel("swap gap (MASE swapped - intact)")
_ax.set_title("Smoke: real checkpoints on planted panels")
_ax.legend(fontsize=8)
_fig.tight_layout()
_fig
'''

async with cm.get_context() as ctx:
    mode = "PLOT"
    if mode == "LOAD":
        cid = ctx.create_cell(LOAD_CODE, hide_code=False)
        print("created", cid)
        ctx.run_cell(cid)
    elif mode == "GRID":
        cid = ctx.create_cell(GRID_CODE, hide_code=False)
        print("created", cid)
        ctx.run_cell(cid)
    elif mode == "PLOT":
        cid = ctx.create_cell(PLOT_CODE, hide_code=False)
        print("created", cid)
        ctx.run_cell(cid)
print("queued")
