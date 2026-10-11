"""Cross-window probe tests — mocks only. Windows = planted panels, distinct seeds."""
from tsfm_audit.adapters import naive_last_value, oracle_lagged
from tsfm_audit.harness import run_probe_windows
from tsfm_audit.planted import PlantedConfig, generate_planted_panel


def _windows(beta=1.0, n=4):
    out = []
    for s in range(n):
        p = generate_planted_panel(beta, PlantedConfig(n_series=1, seed=100 + s))
        out.append({
            "Y_past": p["Y_past"][0], "X_past": p["X_past"][0],
            "X_future": p["X_future"][0], "Y_future": p["Y_future"][0],
        })
    return out


def test_windows_blind_gap_zero():
    r = run_probe_windows("naive", "t", _windows(1.0), naive_last_value, seed=0)
    assert abs(r.swap_gap) < 1e-9


def test_windows_oracle_gap_positive():
    r = run_probe_windows("oracle", "t", _windows(1.0), oracle_lagged(beta=1.0), seed=0)
    assert r.swap_gap > 0.05


def test_windows_ragged_context_truncates():
    ws = _windows(1.0)
    ws[0]["Y_past"] = ws[0]["Y_past"][:64]
    ws[0]["X_past"] = ws[0]["X_past"][:64, :]
    r = run_probe_windows("oracle", "t", ws, oracle_lagged(beta=1.0), seed=0)
    assert r.swap_gap > 0.05


def test_windows_rejects_single():
    import pytest

    with pytest.raises(ValueError):
        run_probe_windows("m", "t", _windows(1.0, n=1), naive_last_value, seed=0)
