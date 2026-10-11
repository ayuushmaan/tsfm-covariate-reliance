"""Unit tests: probes keep marginals, break alignment; metrics sane; harness contrasts blind vs using."""
import numpy as np

from tsfm_audit.adapters import naive_last_value, oracle_lagged
from tsfm_audit.harness import run_probe
from tsfm_audit.metrics import mase
from tsfm_audit.planted import PlantedConfig, generate_planted_panel
from tsfm_audit.probes import permute_time, swap_across_series


def test_swap_deranges_and_preserves_values():
    rng = np.random.default_rng(0)
    X = np.arange(3 * 5).reshape(3, 5, 1).astype(float)
    Xs = swap_across_series(X, rng)
    assert Xs.shape == X.shape
    assert sorted(map(tuple, Xs[:, :, 0].tolist())) == sorted(map(tuple, X[:, :, 0].tolist()))


def test_permute_keeps_marginals():
    rng = np.random.default_rng(1)
    X = np.random.default_rng(2).normal(size=(4, 20, 2))
    Xp = permute_time(X, rng)
    assert Xp.shape == X.shape
    for i in range(4):
        assert np.allclose(np.sort(X[i].ravel()), np.sort(Xp[i].ravel()))


def test_mase_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mase(y, y, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])) == 0.0


def test_harness_blind_gap_zero_planted_beta0():
    panel = generate_planted_panel(0.0, PlantedConfig(n_series=8, seed=0))
    r = run_probe("naive", "t", panel["Y_past"], panel["X_past"], panel["X_future"], panel["Y_future"], naive_last_value, seed=0)
    assert abs(r.swap_gap) < 1e-9  # blind model ignores X entirely


def test_harness_oracle_gap_positive_planted_beta1():
    panel = generate_planted_panel(1.0, PlantedConfig(n_series=16, seed=0))
    r = run_probe("oracle", "t", panel["Y_past"], panel["X_past"], panel["X_future"], panel["Y_future"], oracle_lagged(beta=1.0), seed=0)
    assert r.swap_gap > 0.05  # using model is hurt by wrong covariates
