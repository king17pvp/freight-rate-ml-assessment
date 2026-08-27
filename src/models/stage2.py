from __future__ import annotations

import lightgbm as lgb
import numpy as np

RANDOM_STATE = 0
CATEGORICAL = ["equipment"]

LGB_PARAMS_STAGE2 = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.02,
    num_leaves=7,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=3000,
    random_state=RANDOM_STATE,
    verbose=-1,
)
STAGE2_EARLY_STOPPING_PATIENCE = 300


def fit_stage2_lgb(X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(**LGB_PARAMS_STAGE2)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(STAGE2_EARLY_STOPPING_PATIENCE, verbose=False)],
    )
    return model


def combine_predictions(offset_pred_log, daily_level, distance):
    return np.exp(offset_pred_log) * daily_level * distance
