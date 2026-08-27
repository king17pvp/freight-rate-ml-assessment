import numpy as np
import pandas as pd

from features.geo import add_shipment_geo_date_features, haversine_miles
from features.lane_encoding import (
    K_CITY,
    K_LANE,
    apply_lane_target_encoder_shrink,
    fit_lane_target_encoder_shrink,
)
from features.market import build_daily_index, ets_forecast


def test_haversine_known_city_pair():
    # New York City -> Los Angeles, published great-circle distance ~2,445-2,451 miles.
    nyc = (40.7128, -74.0060)
    la = (34.0522, -118.2437)
    distance = haversine_miles(*nyc, *la)
    assert 2400 < distance < 2500


def test_haversine_same_point_is_zero():
    assert haversine_miles(38.0, -76.0, 38.0, -76.0) == 0.0


def test_add_shipment_geo_date_features_adds_expected_columns():
    df = pd.DataFrame({
        "pickup_lat": [40.7128], "pickup_lon": [-74.0060],
        "delivery_lat": [34.0522], "delivery_lon": [-118.2437],
        "distance": [2500.0],
        "equipment": ["Dry Van"],
        "date": pd.to_datetime(["2025-06-15"]),
    })
    out = add_shipment_geo_date_features(df)
    expected_cols = {
        "log_distance", "haversine", "circuity", "lon_delta",
        "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
        "is_weekend", "days_since_start", "days_to_christmas", "days_to_new_year",
    }
    assert expected_cols.issubset(out.columns)
    assert str(out["equipment"].dtype) == "category"
    assert out["log_distance"].iloc[0] == np.log1p(2500.0)
    assert out["days_since_start"].iloc[0] == (pd.Timestamp("2025-06-15") - pd.Timestamp("2025-01-01")).days
    assert out["is_weekend"].iloc[0] == 1  # 2025-06-15 is a Sunday


def _lane_training_frame():
    # Two lanes, 3 obs each, with different means -- enough to see shrinkage act.
    return pd.DataFrame({
        "pickup": ["Richmond", "Richmond", "Richmond", "Denver", "Denver", "Denver"],
        "delivery": ["Baltimore", "Baltimore", "Baltimore", "Reno", "Reno", "Reno"],
    })


def test_lane_encoder_novel_lane_collapses_to_city_prior():
    train_df = _lane_training_frame()
    target = pd.Series([6.0, 6.2, 5.8, 7.0, 7.4, 6.6], index=train_df.index)  # log-rate-ish
    encoder = fit_lane_target_encoder_shrink(train_df, target, k_lane=K_LANE, k_city=K_CITY)

    novel = pd.DataFrame({"pickup": ["Richmond"], "delivery": ["Nowhere"]})  # unseen lane
    encoded = apply_lane_target_encoder_shrink(novel, encoder)

    pickup_shrunk = encoder["pickup_shrunk"]["Richmond"]
    delivery_shrunk_for_novel_city = encoder["global"]  # "Nowhere" was never a delivery city
    expected_city_prior = (pickup_shrunk + delivery_shrunk_for_novel_city) / 2.0
    assert abs(encoded.iloc[0] - expected_city_prior) < 1e-9


def test_lane_encoder_seen_lane_uses_lane_mean_shrunk_toward_city_prior():
    train_df = _lane_training_frame()
    target = pd.Series([6.0, 6.2, 5.8, 7.0, 7.4, 6.6], index=train_df.index)
    encoder = fit_lane_target_encoder_shrink(train_df, target, k_lane=K_LANE, k_city=K_CITY)

    seen = pd.DataFrame({"pickup": ["Richmond"], "delivery": ["Baltimore"]})
    encoded = apply_lane_target_encoder_shrink(seen, encoder)
    # With k_lane=60 and only 3 lane observations, the encoding should sit strictly between
    # the raw lane mean and the city prior -- heavy shrinkage toward the prior, not equal to
    # either extreme.
    raw_lane_mean = target.iloc[:3].mean()
    city_prior = (encoder["pickup_shrunk"]["Richmond"] + encoder["delivery_shrunk"]["Baltimore"]) / 2.0
    low, high = sorted([raw_lane_mean, city_prior])
    assert low <= encoded.iloc[0] <= high


def _daily_history(periods=60, base=2.0):
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    # mild upward trend + weekly wobble, deterministic (no randomness) so the test is stable.
    dow = dates.dayofweek.to_numpy()
    values = base + 0.01 * np.arange(periods) + 0.05 * np.sin(2 * np.pi * dow / 7)
    return pd.Series(values, index=dates)


def test_build_daily_index_has_full_daily_calendar_and_no_nans():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-03"]),  # 2025-01-02 missing
        "posted_rate": [200.0, 220.0, 300.0],
        "distance": [100.0, 100.0, 100.0],
    })
    daily = build_daily_index(df)
    assert list(daily.index) == list(pd.date_range("2025-01-01", "2025-01-03", freq="D"))
    assert not daily["rpm_median"].isna().any()  # the gap day is interpolated, not left NaN


def test_ets_forecast_returns_requested_horizon_length():
    history = _daily_history(periods=60)
    forecast = ets_forecast(history, horizon=14)
    assert isinstance(forecast, np.ndarray)
    assert len(forecast) == 14
    assert np.isfinite(forecast).all()
