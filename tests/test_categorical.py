"""Categorical path tests — mocks only, zero downloads. Mirrors rossmann structure."""
import numpy as np
import pandas as pd

from tsfm_audit.adapters import arrays_to_long_df, make_bolt_predict, make_chronos2_predict
from tsfm_audit.fev_convert import window_to_arrays
from tsfm_audit.harness import run_probe


class FakeWindow:
    """Mimics fev EvaluationWindow: numeric known + numeric past + str/None statics."""

    known_dynamic_columns = ["Promo", "Open"]
    past_dynamic_columns = ["Customers"]
    static_columns = ["Assortment", "StoreType", "PromoInterval"]
    target = "Sales"
    horizon = 6

    def __init__(self, n=6, t=30, seed=0):
        rng = np.random.default_rng(seed)
        self.past_rows, self.fut_rows, self.gt_rows = [], [], []
        for i in range(n):
            promo = rng.integers(0, 2, t + 6).astype(float)
            sales = 5000 + 300 * promo + rng.normal(0, 50, t + 6)
            self.past_rows.append({
                "id": f"s{i}", "timestamp": list(range(t)), "Sales": sales[:t],
                "Promo": promo[:t], "Open": np.ones(t),
                "Customers": sales[:t] / 10, "Assortment": "a" if i % 2 == 0 else "b",
                "StoreType": "c", "PromoInterval": None,
            })
            self.fut_rows.append({
                "id": f"s{i}", "timestamp": list(range(t, t + 6)),
                "Promo": promo[t:], "Open": np.ones(6),
            })
            self.gt_rows.append({"id": f"s{i}", "Sales": sales[t:]})

    def get_input_data(self):
        return self.past_rows, self.fut_rows

    def get_ground_truth(self):
        return FakeDS(self.gt_rows, ["id", "Sales"])


class FakeDS:
    def __init__(self, rows, cols):
        self._rows = rows
        self.column_names = cols

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


class RecordingC2:
    """Fake Chronos-2: uses decoded Promo strings/floats; records frames."""

    def __init__(self):
        self.frames = []

    def predict_df(self, df, future_df=None, prediction_length=None, **kw):
        self.frames.append((df, future_df))
        rows = []
        for sid, g in df.groupby("id"):
            f = future_df[future_df["id"] == sid].sort_values("timestamp")
            last = float(g["target"].iloc[-1])
            pv = f["Promo"].to_numpy(dtype=float)
            for t, v in zip(f["timestamp"].to_numpy(), last + 300 * pv[:prediction_length]):
                rows.append({"id": sid, "timestamp": t, "target_name": "target", "predictions": float(v)})
        return pd.DataFrame(rows)


class RecordingBolt:
    def __init__(self):
        self.frames = []

    def predict_df(self, df, prediction_length=None, **kw):
        self.frames.append(df)
        rows = []
        for sid, g in df.groupby("id"):
            for k in range(prediction_length):
                rows.append({"id": sid, "timestamp": k, "target_name": "target",
                             "predictions": float(g["target"].iloc[-1])})
        return pd.DataFrame(rows)


def _convert():
    return window_to_arrays(FakeWindow(), max_context=None)


def test_factorize_and_names():
    a = _convert()
    assert a["cat_maps"]["Assortment"] == ["a", "b"]
    assert a["cat_maps"]["StoreType"] == ["c"]
    assert a["col_names"][:2] == ["Promo", "Open"]
    assert a["known_names"] == ["Promo", "Open"]
    assert a["X_past"].shape == (6, 30, 6)


def test_frames_carry_strings_and_known_only_future():
    a = _convert()
    c2, bolt = RecordingC2(), RecordingBolt()
    pc2 = make_chronos2_predict(c2, ids=a["ids"], col_names=a["col_names"],
                                known_names=a["known_names"], cat_maps=a["cat_maps"])
    pbo = make_bolt_predict(bolt, ids=a["ids"], col_names=a["col_names"],
                            known_names=a["known_names"], cat_maps=a["cat_maps"])
    pc2(a["Y_past"], a["X_past"], a["X_future"])
    pbo(a["Y_past"], a["X_past"], a["X_future"])
    ctx, fut = c2.frames[0]
    assert set(fut.columns) == {"id", "timestamp", "Promo", "Open"}
    assert ctx["Assortment"].dtype == object
    assert set(ctx["Assortment"].unique()) == {"a", "b"}
    assert ctx["StoreType"].unique().tolist() == ["c"]
    assert set(bolt.frames[0].columns) == {"id", "timestamp", "target"}


def test_probe_contrast_with_categoricals():
    a = _convert()
    kw = dict(Y_past=a["Y_past"], X_past=a["X_past"], X_future=a["X_future"], Y_future=a["Y_future"])
    meta = dict(ids=a["ids"], col_names=a["col_names"], known_names=a["known_names"], cat_maps=a["cat_maps"])
    r_c2 = run_probe("c2", "t", **kw, predict=make_chronos2_predict(RecordingC2(), **meta), seed=0)
    r_bo = run_probe("bolt", "t", **kw, predict=make_bolt_predict(RecordingBolt(), **meta), seed=0)
    assert r_c2.swap_gap > 0.01
    assert abs(r_bo.swap_gap) < 1e-9
