import numpy as np

from eval import mae, rmse, smape


def test_mae_basic():
    y_true = [100.0, 200.0, 300.0]
    y_pred = [110.0, 190.0, 320.0]
    assert abs(mae(y_true, y_pred) - (10 + 10 + 20) / 3) < 1e-9


def test_rmse_basic():
    y_true = [0.0, 0.0]
    y_pred = [3.0, 4.0]
    assert abs(rmse(y_true, y_pred) - np.sqrt(12.5)) < 1e-9


def test_smape_perfect_prediction_is_zero():
    y_true = [100.0, 250.0, 30.0]
    assert smape(y_true, y_true) == 0.0


def test_smape_is_a_percentage_and_symmetric():
    y_true = [100.0]
    y_pred = [150.0]
    assert abs(smape(y_true, y_pred) - 40.0) < 1e-9
    assert abs(smape(y_pred, y_true) - 40.0) < 1e-9


def test_smape_handles_both_zero_without_nan():
    y_true = [0.0, 5.0]
    y_pred = [0.0, 5.0]
    result = smape(y_true, y_pred)
    assert not np.isnan(result)
    assert result == 0.0
