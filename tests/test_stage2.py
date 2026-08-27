import numpy as np
import pandas as pd

from models.stage2 import LGB_PARAMS_STAGE2, combine_predictions, fit_stage2_lgb


def _synthetic_stage2_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    equipment = pd.Categorical(rng.choice(["Dry Van", "Reefer", "Flatbed"], size=n))
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.05, size=n)
    y = 0.3 * x1 - 0.2 * x2 + noise
    X = pd.DataFrame({"x1": x1, "x2": x2, "equipment": equipment})
    return X, pd.Series(y)


def test_fit_stage2_lgb_produces_a_fitted_model_with_best_iteration():
    X, y = _synthetic_stage2_frame(n=400)
    split = 300
    model = fit_stage2_lgb(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
    assert model.best_iteration_ is not None
    assert model.best_iteration_ > 0
    preds = model.predict(X.iloc[split:], num_iteration=model.best_iteration_)
    assert len(preds) == len(X) - split
    assert np.isfinite(preds).all()


def test_combine_predictions_matches_exp_times_level_times_distance():
    offset_pred_log = np.array([0.0, np.log(2.0)])
    daily_level = np.array([1.5, 1.5])
    distance = np.array([100.0, 100.0])
    result = combine_predictions(offset_pred_log, daily_level, distance)
    expected = np.array([1.0 * 1.5 * 100.0, 2.0 * 1.5 * 100.0])
    assert np.allclose(result, expected)


def test_lgb_params_stage2_has_expected_config():
    assert LGB_PARAMS_STAGE2["learning_rate"] == 0.02
    assert LGB_PARAMS_STAGE2["num_leaves"] == 7
    assert LGB_PARAMS_STAGE2["min_child_samples"] == 30
