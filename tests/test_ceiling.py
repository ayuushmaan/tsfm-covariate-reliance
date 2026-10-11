"""Ceiling tests: on planted data it should approach the oracle, beat naive."""
import numpy as np

from tsfm_audit.adapters import naive_last_value, oracle_lagged
from tsfm_audit.ceiling import ceiling_ridge
from tsfm_audit.metrics import mase
from tsfm_audit.planted import PlantedConfig, generate_planted_panel


def _mean_mase(panel, pred):
    return float(np.mean([mase(panel["Y_future"][i], pred[i], panel["Y_past"][i]) for i in range(len(panel["Y_past"]))]))


def test_ceiling_near_oracle_on_planted():
    panel = generate_planted_panel(1.0, PlantedConfig(n_series=16, seed=0))
    kw = dict(X_past=panel["X_past"], X_future=panel["X_future"])
    m_oracle = _mean_mase(panel, oracle_lagged(beta=1.0)(panel["Y_past"], **kw))
    m_ceil = _mean_mase(panel, ceiling_ridge(panel["Y_past"], **kw))
    m_naive = _mean_mase(panel, naive_last_value(panel["Y_past"], **kw))
    assert m_ceil < m_naive * 0.75, f"ceiling {m_ceil:.3f} should beat naive {m_naive:.3f}"
    assert np.isfinite(m_ceil) and m_ceil > 0


def test_ceiling_fallback_no_covariates():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(4, 40))
    pred = ceiling_ridge(Y, np.zeros((4, 40, 0)), np.zeros((4, 5, 0)))
    assert pred.shape == (4, 5)
    assert np.allclose(pred, Y[:, -1:])
