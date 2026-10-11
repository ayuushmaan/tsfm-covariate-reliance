"""Degenerate-scale series return NaN (skipped by nanmean); healthy values unchanged."""
import numpy as np

from tsfm_audit.metrics import mase


def test_zero_scale_returns_nan():
    y = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 0.0])
    assert np.isnan(mase(np.array([1.0, 2.0]), np.array([1.5, 2.5]), np.zeros(10)))


def test_healthy_value_unchanged():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2])
    train = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    scale = np.mean(np.abs(train[1:] - train[:-1]))
    assert mase(y_true, y_pred, train) == float(np.mean(np.abs(y_true - y_pred)) / scale)
