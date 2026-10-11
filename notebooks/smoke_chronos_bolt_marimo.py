# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "chronos-forecasting",
#     "fev",
#     "pandas",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///
"""Smoke: Chronos-2 vs Bolt on the covariate-reliance probe — GPU box only.

Open on your marimo/Modal GPU (nothing runs on the laptop):
    marimo run notebooks/smoke_chronos_bolt_marimo.py -- --max-tasks 2
or as editable notebook:
    marimo edit notebooks/smoke_chronos_bolt_marimo.py

What it does:
  1. checks GPU, loads amazon/chronos-2 (120M) + amazon/chronos-bolt-small (48M)
  2. runs intact-vs-swapped probe on PLANTED panels (beta grid) through the REAL checkpoints
  3. lists fev-bench-mini covariate tasks + inspects first window schema (no modeling yet)
Expectation: chronos2 swap_gap rises with beta; bolt gap stays ~0.
"""
import marimo as mo

app = mo.App(title="TSFM smoke: chronos2 vs bolt")


@app.cell
def setup():
    import sys

    import torch

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dev_name = torch.cuda.get_device_name(0) if has_cuda else "cpu"
    print(f"python={sys.version.split()[0]} device={device} ({dev_name})")
    max_tasks = 2
    betas = [0.0, 0.5, 1.0]
    return betas, device, max_tasks, torch


@app.cell
def load_models(device):
    # HEAVY CELL — GPU box only. Downloads ~170M checkpoints here, never locally.
    from tsfm_audit.adapters import load_bolt, load_chronos2, make_bolt_predict, make_chronos2_predict

    chronos2_pipe = load_chronos2(device)
    bolt_pipe = load_bolt(device)
    chronos2_predict = make_chronos2_predict(chronos2_pipe)
    bolt_predict = make_bolt_predict(bolt_pipe)
    print("checkpoints loaded")
    return bolt_predict, chronos2_predict


@app.cell
def planted_probe(betas, bolt_predict, chronos2_predict):
    # Real checkpoints on synthetic panels with known effect size — no fev schema risk.
    from tsfm_audit.harness import run_probe
    from tsfm_audit.planted import PlantedConfig, generate_planted_panel

    smoke_rows = []
    for _beta in betas:
        _panel = generate_planted_panel(_beta, PlantedConfig(n_series=16, seed=0))
        _kw = dict(
            Y_past=_panel["Y_past"],
            X_past=_panel["X_past"],
            X_future=_panel["X_future"],
            Y_future=_panel["Y_future"],
        )
        smoke_rows.append(
            run_probe("chronos2", f"planted_beta{_beta:g}", **_kw, predict=chronos2_predict, seed=0).to_dict()
        )
        smoke_rows.append(
            run_probe("bolt", f"planted_beta{_beta:g}", **_kw, predict=bolt_predict, seed=0).to_dict()
        )
        print(f"beta={_beta:.1f} chronos2_gap={smoke_rows[-2]['swap_gap']:+.4f} bolt_gap={smoke_rows[-1]['swap_gap']:+.4f}")
    return (smoke_rows,)


@app.cell
def plot_smoke(betas, smoke_rows):
    import matplotlib.pyplot as plt

    _c = [r["swap_gap"] for r in smoke_rows if r["model"] == "chronos2"]
    _b = [r["swap_gap"] for r in smoke_rows if r["model"] == "bolt"]
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.plot(betas, _c, "s-", label="chronos2 (expect rising)")
    _ax.plot(betas, _b, "o--", label="bolt control (expect ~0)")
    _ax.set_xlabel("planted effect β")
    _ax.set_ylabel("swap gap (MASE swapped − intact)")
    _ax.set_title("Smoke: real checkpoints on planted panels")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def inspect_fev_mini(max_tasks):
    # No modeling here — just list covariate tasks + dump first window schema
    # so the full probe loop can be pinned down interactively in one pass.
    from tsfm_audit.adapters import FEV_MINI_YAML

    from fev import Benchmark

    _bench = Benchmark.from_yaml(FEV_MINI_YAML)
    _cov = [
        t
        for t in _bench.tasks
        if getattr(t, "known_dynamic_columns", []) or getattr(t, "past_dynamic_columns", [])
    ]
    print(f"mini tasks={len(_bench.tasks)} with_covariates={len(_cov)}")
    for _t in _cov[:max_tasks]:
        print(f"- {_t.task_name} horizon={_t.horizon} known={_t.known_dynamic_columns} past={_t.past_dynamic_columns}")
    _w = _cov[0].get_window(0)
    _past, _future = _w.get_input_data()
    print(type(_past), type(_future))
    print(_past[:2] if hasattr(_past, "iloc") else str(_past)[:500])
    return


if __name__ == "__main__":
    app.run()
