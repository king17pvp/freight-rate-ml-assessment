from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.geo import add_shipment_geo_date_features
from features.lane_encoding import (
    K_CITY,
    K_LANE,
    apply_lane_target_encoder_shrink,
    fit_lane_target_encoder_shrink,
)
from features.market import build_daily_index, ets_forecast
from models.stage2 import CATEGORICAL, LGB_PARAMS_STAGE2, combine_predictions, fit_stage2_lgb

NUMERIC_BASE = [
    "distance", "log_distance", "weight", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "circuity", "lon_delta", "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_weekend", "days_since_start", "days_to_christmas", "days_to_new_year", "lane_target_enc",
]
FEATURE_COLS = NUMERIC_BASE + CATEGORICAL


def _fit_shipment_feature_encoder(train_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit geo/date features + the shrinkage lane encoder on `train_raw`, apply to itself."""
    train_feat = add_shipment_geo_date_features(train_raw)
    y_train_log = np.log1p(train_raw["posted_rate"])
    encoder = fit_lane_target_encoder_shrink(train_feat, y_train_log, k_lane=K_LANE, k_city=K_CITY)
    train_feat["lane_target_enc"] = apply_lane_target_encoder_shrink(train_feat, encoder)
    return train_feat[FEATURE_COLS], encoder


def _apply_shipment_features(raw_df: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    """Apply an already-fit encoder to a new frame (validation/holdout/December)."""
    feat = add_shipment_geo_date_features(raw_df)
    feat["lane_target_enc"] = apply_lane_target_encoder_shrink(feat, encoder)
    return feat[FEATURE_COLS]


def _fit_city_coordinates(train_raw: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Every city has exactly one fixed (lat, lon) across the dataset, whether it appears as
    a pickup or a delivery -- used to backfill coordinates for frames that omit them (see
    `_fill_missing_coordinates`).
    """
    pickup_coords = train_raw[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    delivery_coords = train_raw[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    coords = pd.concat([pickup_coords, delivery_coords]).drop_duplicates(subset="city").set_index("city")
    return {city: (row["lat"], row["lon"]) for city, row in coords.iterrows()}


def _fill_missing_coordinates(df: pd.DataFrame, city_coords: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """december-chart-inputs.csv has no pickup_lat/pickup_lon/delivery_lat/delivery_lon
    columns at all (only the original seven: pickup,delivery,distance,equipment,weight,date,
    predicted_rate) -- look coordinates up by city name instead of requiring the caller to
    supply them for an already-seen city.
    """
    out = df.copy()
    if "pickup_lat" not in out.columns:
        out["pickup_lat"] = out["pickup"].map(lambda city: city_coords[city][0])
        out["pickup_lon"] = out["pickup"].map(lambda city: city_coords[city][1])
    if "delivery_lat" not in out.columns:
        out["delivery_lat"] = out["delivery"].map(lambda city: city_coords[city][0])
        out["delivery_lon"] = out["delivery"].map(lambda city: city_coords[city][1])
    return out


def _forecast_daily_level(daily_train: pd.Series, target_dates: pd.Series) -> pd.Series:
    """Stage 1: extend `daily_train` far enough to cover every date in `target_dates`, then
    map each target date to its forecasted level. Horizon is always measured from
    `daily_train`'s last observed date (not from `target_dates`'s own start) so the forecast
    is a deterministic function of training history + how far out we need to go -- the same
    values come out whether `target_dates` starts immediately after training (validation.csv)
    or with a gap (december-chart-inputs.csv alone, without validation.csv extending it).
    """
    horizon = (target_dates.max() - daily_train.index[-1]).days
    forecast_index = pd.date_range(daily_train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = pd.Series(ets_forecast(daily_train, horizon), index=forecast_index)
    return target_dates.map(forecast)


@dataclass
class FittedTwoStageModel:
    lane_encoder: dict
    daily_train: pd.Series
    model: Any
    city_coords: dict[str, tuple[float, float]]


def fit(train_df: pd.DataFrame) -> FittedTwoStageModel:
    clean = train_df.copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean["weight"] = clean["weight"].abs()

    daily_train = build_daily_index(clean)["rpm_median"]

    # Early-stopping tail (last 2 months) selects n_estimators; discarded after that.
    es_tail_start = clean["date"].max() - pd.DateOffset(months=2) + pd.Timedelta(days=1)
    es_fit = clean[clean["date"] < es_tail_start]
    es_val = clean[clean["date"] >= es_tail_start]

    X_es_fit, es_encoder = _fit_shipment_feature_encoder(es_fit)
    X_es_val = _apply_shipment_features(es_val, es_encoder)
    es_fit_level = es_fit["date"].map(daily_train).to_numpy()
    es_val_level = es_val["date"].map(daily_train).to_numpy()
    y_es_fit = np.log(es_fit["posted_rate"].to_numpy() / (es_fit_level * es_fit["distance"].to_numpy()))
    y_es_val = np.log(es_val["posted_rate"].to_numpy() / (es_val_level * es_val["distance"].to_numpy()))

    es_model = fit_stage2_lgb(X_es_fit, y_es_fit, X_es_val, y_es_val)
    final_n_estimators = es_model.best_iteration_

    # Final model: fit on the entire pool, fixed n_estimators, no further early stopping.
    X_train, final_encoder = _fit_shipment_feature_encoder(clean)
    train_level = clean["date"].map(daily_train).to_numpy()
    y_train = np.log(clean["posted_rate"].to_numpy() / (train_level * clean["distance"].to_numpy()))

    final_params = dict(LGB_PARAMS_STAGE2, n_estimators=final_n_estimators)
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL)

    city_coords = _fit_city_coordinates(clean)

    return FittedTwoStageModel(lane_encoder=final_encoder, daily_train=daily_train, model=model, city_coords=city_coords)


def predict(fitted: FittedTwoStageModel, raw_df: pd.DataFrame) -> np.ndarray:
    apply_df = raw_df.copy()
    apply_df["date"] = pd.to_datetime(apply_df["date"])
    apply_df["weight"] = apply_df["weight"].abs()
    apply_df = _fill_missing_coordinates(apply_df, fitted.city_coords)

    X_apply = _apply_shipment_features(apply_df, fitted.lane_encoder)
    daily_level_forecast = _forecast_daily_level(fitted.daily_train, apply_df["date"])
    offset_pred = fitted.model.predict(X_apply)
    return combine_predictions(offset_pred, daily_level_forecast.to_numpy(), apply_df["distance"].to_numpy())
