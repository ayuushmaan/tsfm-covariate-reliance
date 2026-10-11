"""GPU-free demo: dose-response on planted signals.

Compares a covariate-blind baseline (naive) vs a covariate-using oracle
across beta grid. Proves the harness detects reliance when it exists and
returns ~0 gap when it doesn't — the Figure-2 miniature.

Usage:
    python experiments/run_planted_demo.py --betas 0 0.2 0.4 0.6 0.8 1.0 --n-series 32 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tsfm_audit.adapters import naive_last_value, oracle_lagged
from tsfm_audit.ceiling import ceiling_ridge
from tsfm_audit.harness import run_probe
from tsfm_audit.metrics import mase
from tsfm_audit.planted import PlantedConfig, generate_planted_panel


def eval_gap(panel: dict, predict, model: str, task: str, seed: int) -> dict:
    r = run_probe(model, task, panel["Y_past"], panel["X_past"], panel["X_future"], panel["Y_future"], predict, seed)
    return r.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=float, nargs="+", default=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--n-series", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/demo_planted.json")
    args = ap.parse_args()

    rows = []
    for beta in args.betas:
        cfg = PlantedConfig(n_series=args.n_series, seed=args.seed)
        panel = generate_planted_panel(beta, cfg)
        task = f"planted_beta{beta:g}"
        rows.append(eval_gap(panel, naive_last_value, "naive_blind", task, args.seed))
        rows.append(eval_gap(panel, oracle_lagged(beta=1.0), "oracle_uses_covariates", task, args.seed))
        # ceiling reference: ridge MASE vs naive MASE
        ceil_pred = ceiling_ridge(panel["Y_past"], panel["X_past"], panel["X_future"])
        ceil_mase = float(np.mean([mase(panel["Y_future"][i], ceil_pred[i], panel["Y_past"][i]) for i in range(len(panel["Y_past"]))]))
        rows[-1]["ceiling_mase"] = ceil_mase
        rows[-2]["ceiling_mase"] = ceil_mase
        print(f"beta={beta:4.1f} blind_gap={rows[-2]['swap_gap']:+.4f} oracle_gap={rows[-1]['swap_gap']:+.4f} ceiling_mase={ceil_mase:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out} ({len(rows)} rows)")

    # figure (optional matplotlib)
    try:
        import matplotlib.pyplot as plt

        betas = args.betas
        blind = [r["swap_gap"] for r in rows if r["model"] == "naive_blind"]
        oracle = [r["swap_gap"] for r in rows if r["model"] == "oracle_uses_covariates"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(betas, blind, "o--", label="blind (naive) — expect ~0")
        ax.plot(betas, oracle, "s-", label="uses covariates (oracle)")
        ax.set_xlabel("planted effect β")
        ax.set_ylabel("swap gap (MASE swapped − intact)")
        ax.set_title("Planted-signal dose-response (demo)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figpath = out.parent / "demo_figure.png"
        fig.savefig(figpath, dpi=150)
        print(f"wrote {figpath}")
    except ImportError:
        print("matplotlib not installed; skipping figure")


if __name__ == "__main__":
    main()
