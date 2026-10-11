"""fev window -> harness arrays. GPU-box side (duck-typed, no heavy imports).

Conventions:
- Columns: known-dynamic + past-dynamic (+ static as constants), real names kept.
- Categoricals (strings/None, e.g. rossmann Assortment/StoreType/PromoInterval)
  are factorized to integer codes; `cat_maps` carries {col: [categories]} so
  adapters can decode back to strings (Chronos-2 encodes them itself).
- X_future holds true values for known cols, NaN for past/static; adapters
  drop non-declared cols from future_df -> true past-only semantics.
- Gaps (store closures etc.) are forward-filled, then zero/mode-filled.
"""
from __future__ import annotations

import numpy as np


def _seq(row: dict, col: str) -> np.ndarray:
    return np.asarray(row[col])


def _ffill(a: np.ndarray, seed_row: np.ndarray | None = None) -> np.ndarray:
    """Forward-fill NaNs along axis 0 (per column), then zero-fill. Documents real-world gaps."""
    out = a.copy().astype(float)
    if seed_row is not None:
        first = out[0]
        out[0] = np.where(np.isnan(first), seed_row, first)
    for t in range(1, out.shape[0]):
        prev, cur = out[t - 1], out[t]
        out[t] = np.where(np.isnan(cur), prev, cur)
    return np.where(np.isnan(out), 0.0, out)


def _is_cat(values: list) -> bool:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            return True
        try:
            float(v)
        except (TypeError, ValueError):
            return True
        return False
    return False


def window_to_arrays(window, max_context: int | None = 512) -> dict:
    known = list(getattr(window, "known_dynamic_columns", []) or [])
    past_cols = list(getattr(window, "past_dynamic_columns", []) or [])
    static = list(getattr(window, "static_columns", []) or [])
    tcols = getattr(window, "target_columns", None) or getattr(window, "target", "target")
    target = tcols[0] if isinstance(tcols, (list, tuple)) else tcols

    past_ds, fut_ds = window.get_input_data()
    covs = known + past_cols + static
    n_known = len(known)

    rows = list(past_ds)
    lens = [len(np.asarray(r[target])) for r in rows]
    T_use = min(lens)
    if max_context is not None:
        T_use = min(T_use, max_context)
    H = window.horizon

    # pass 1: detect categoricals, collect global categories
    cat_maps: dict[str, list[str]] = {}
    is_cat_col: dict[str, bool] = {}
    for c in covs:
        sample: list = []
        for r in rows[: min(50, len(rows))]:
            v = r[c]
            a = np.asarray(v)
            sample.extend(a.tolist() if a.shape else [a.item()])
            if len(sample) > 200:
                break
        if _is_cat(sample):
            cats = sorted({str(v) for v in sample if v is not None})
            cat_maps[c] = cats or ["MISSING"]
            is_cat_col[c] = True
        else:
            is_cat_col[c] = False

    def encode(values: np.ndarray, col: str) -> np.ndarray:
        if not is_cat_col[col]:
            return np.array([np.nan if v is None else float(v) for v in values.tolist()], dtype=float)
        cats = cat_maps[col]
        idx = {s: i for i, s in enumerate(cats)}
        return np.array(
            [np.nan if v is None else float(idx.get(str(v), 0)) for v in values.tolist()], dtype=float
        )

    Yp, Xp, Xf, Yf, ids, p_ts, f_ts = [], [], [], [], [], [], []
    gt = window.get_ground_truth()
    gt_cols = gt.column_names if hasattr(gt, "column_names") else []
    gt_target = target if target in gt_cols else (gt_cols[-1] if gt_cols else target)
    gt_by_id = {}
    if hasattr(gt, "__getitem__") and gt_target in gt_cols:
        for r in gt:
            gt_by_id[str(r["id"])] = np.asarray(r[gt_target], dtype=float)
    else:
        raise TypeError(f"unexpected ground-truth format: {type(gt)} cols={gt_cols}")

    for i, row in enumerate(rows):
        frow = fut_ds[i]
        y = _seq(row, target).astype(float)[-T_use:]
        cols_past, cols_fut = [], []
        for c in covs:
            if c in static:
                a = np.asarray(row[c])
                v = a.item() if not a.shape else a.tolist()
                vals = [v] * T_use
                cols_past.append(encode(np.asarray(vals, dtype=object), c))
                cols_fut.append(np.full(H, np.nan))
            elif c in known:
                cols_past.append(encode(_seq(row, c)[-T_use:], c))
                cols_fut.append(encode(_seq(frow, c)[:H], c))
            else:
                cols_past.append(encode(_seq(row, c)[-T_use:], c))
                cols_fut.append(np.full(H, np.nan))
        Xp.append(np.stack(cols_past, axis=1) if cols_past else np.zeros((T_use, 0)))
        Xf.append(np.stack(cols_fut, axis=1) if cols_past else np.zeros((H, 0)))
        Yp.append(y)
        sid = str(row["id"])
        ids.append(sid)
        Yf.append(gt_by_id[sid][:H])
        p_ts.append(np.asarray(row["timestamp"])[-T_use:])
        f_ts.append(np.asarray(frow["timestamp"])[:H])

    Xp_f = [_ffill(a) for a in Xp]
    Xf_f = [_ffill(a, seed_row=Xp_f[i][-1]) for i, a in enumerate(Xf)]
    for a in Xf_f:
        a[:, n_known:] = np.nan  # past/static stay NaN -> adapters drop them from future_df
    Yp_f = [_ffill(y[:, None])[:, 0] for y in Yp]
    return {
        "Y_past": np.stack(Yp_f), "X_past": np.stack(Xp_f),
        "X_future": np.stack(Xf_f),
        "Y_future": np.stack(Yf), "ids": ids, "past_ts": p_ts, "fut_ts": f_ts,
        "covs": covs, "n_known": n_known, "target": target,
        "col_names": covs, "known_names": known, "cat_maps": cat_maps,
    }
