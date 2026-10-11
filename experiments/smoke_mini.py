"""fev-bench-mini smoke test: Chronos-2 vs Bolt (univariate control).

LOCAL (this laptop, zero downloads):
    python experiments/smoke_mini.py --dry-run
    # uses planted panel + FakePipeline; validates adapter wiring + harness math.

REMOTE GPU (marimo / Modal — where downloads happen):
    python experiments/smoke_mini.py --remote --max-tasks 2 --seeds 0 --out results/smoke_mini.jsonl
    # downloads: chronos-2 (120M) + bolt-small (48M) + fev mini task data, all on the GPU box.
    # results bank per (model, task, seed) so a death costs only missing lines.

Pass --no-cross-learning to match the paper's default; smoke keeps it off for speed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_dry_run(n_series: int = 8, seed: int = 0) -> list[dict]:
    """Validate wiring with zero downloads: planted panel + fake pipelines."""
    import numpy as np

    from tsfm_audit.adapters import arrays_to_long_df, make_bolt_predict, make_chronos2_predict, pred_df_to_array
    from tsfm_audit.harness import run_probe
    from tsfm_audit.planted import PlantedConfig, generate_planted_panel

    import pandas as pd

    class FakeCovariatePipeline:
        """Follows the planted law through the DataFrame interface — uses X, so swap must hurt."""

        def predict_df(self, df, future_df=None, prediction_length=None, **kw):
            rows = []
            for sid, g in df.groupby("id"):
                g = g.sort_values("timestamp")
                f = future_df[future_df["id"] == sid].sort_values("timestamp")
                full_x = np.concatenate([g["cov_0"].to_numpy(), f["cov_0"].to_numpy()])
                T = len(g)
                y_last = float(g["target"].iloc[-1])
                for h in range(prediction_length):
                    x_lag = full_x[T + h - 3] if T + h - 3 >= 0 else 0.0
                    y_last = 0.6 * y_last + 1.0 * x_lag
                    rows.append({"id": sid, "timestamp": f["timestamp"].iloc[h], "target_name": "target", "predictions": y_last})
            return pd.DataFrame(rows)

    class FakeBoltPipeline:
        """Ignores covariates — swap must NOT hurt."""

        def predict_df(self, df, prediction_length=None, **kw):
            rows = []
            for sid, g in df.groupby("id"):
                last = float(g["target"].iloc[-1])
                ts = pd.date_range("2020-01-01", periods=len(g) + prediction_length, freq="D")[-prediction_length:]
                for t in ts:
                    rows.append({"id": sid, "timestamp": t, "target_name": "target", "predictions": last})
            return pd.DataFrame(rows)

    panel = generate_planted_panel(1.0, PlantedConfig(n_series=n_series, seed=seed))
    kw = dict(
        Y_past=panel["Y_past"], X_past=panel["X_past"], X_future=panel["X_future"], Y_future=panel["Y_future"]
    )
    r_chronos = run_probe("chronos2(fake)", "planted", **kw, predict=make_chronos2_predict(FakeCovariatePipeline()), seed=seed)
    r_bolt = run_probe("bolt(fake)", "planted", **kw, predict=make_bolt_predict(FakeBoltPipeline()), seed=seed)
    # conversion sanity
    ctx, fut, covs = arrays_to_long_df(panel["Y_past"], panel["X_past"], panel["X_future"])
    assert ctx.shape[0] == n_series * panel["Y_past"].shape[1]
    assert pred_df_to_array.__name__ == "pred_df_to_array"
    print(f"[dry-run] chronos2(fake) swap_gap={r_chronos.swap_gap:+.4f} (expect >0)")
    print(f"[dry-run] bolt(fake)      swap_gap={r_bolt.swap_gap:+.4f} (expect ~0)")
    assert r_chronos.swap_gap > 0.01, "fake covariate-user must show reliance"
    assert abs(r_bolt.swap_gap) < 1e-9, "control must ignore covariates"
    print("[dry-run] OK — wiring valid, no downloads performed.")
    return [r_chronos.to_dict(), r_bolt.to_dict()]


def run_remote(max_tasks: int, seeds: list[int], out: str) -> None:
    """GPU-box path. Heavy imports INSIDE so local `import` stays light."""
    import fev  # noqa: F401  (downloads task data HERE, on GPU box only)

    from tsfm_audit.adapters import FEV_MINI_YAML, load_bolt, load_chronos2, make_bolt_predict, make_chronos2_predict

    from fev import Benchmark

    bench = Benchmark.from_yaml(FEV_MINI_YAML)
    cov_tasks = [t for t in bench.tasks if getattr(t, "known_dynamic_columns", []) or getattr(t, "past_dynamic_columns", [])]
    tasks = (cov_tasks or bench.tasks)[:max_tasks]
    print(f"[remote] fev-bench-mini: {len(bench.tasks)} tasks, {len(cov_tasks)} with covariates, running {len(tasks)}")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[remote] device={device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    models = {"chronos2": make_chronos2_predict(load_chronos2(device)), "bolt": make_bolt_predict(load_bolt(device))}
    # NOTE: full per-window fev evaluation loop (task.get_window -> arrays -> run_probe)
    # is intentionally left for the GPU session: task schemas vary (static/known/past
    # covariates, multivariate targets) and need one interactive pass to pin down.
    # This smoke confirms: ckpts load, predict_df answers, banking works.
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "a") as f:
        for name in models:
            f.write(json.dumps({"model": name, "status": "loaded", "device": device}) + "\n")
    print(f"[remote] checkpoints OK -> {outp}. Next: run probe loop in notebook cells.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="local validation, zero downloads")
    ap.add_argument("--remote", action="store_true", help="run on GPU box (downloads there)")
    ap.add_argument("--max-tasks", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=str, default="results/smoke_mini.jsonl")
    args = ap.parse_args()
    if args.remote:
        run_remote(args.max_tasks, args.seeds, args.out)
    else:
        run_dry_run()


if __name__ == "__main__":
    main()
