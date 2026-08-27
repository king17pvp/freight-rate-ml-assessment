import numpy as np
import pandas as pd

from two_stage_model import fit, predict


def _synthetic_shipment_frame(n_days=150, loads_per_day=4, seed=0):
    """~600-row synthetic frame spanning 150 days -- long enough for a 2-month early-stopping
    tail (fit()) and for ETS's weekly seasonality (features/market.py) to fit without error.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    pickups = ["Richmond", "Denver", "Dallas"]
    deliveries = ["Baltimore", "Reno", "Houston"]
    rows = []
    load_id = 0
    for date in dates:
        for _ in range(loads_per_day):
            load_id += 1
            pu_i, de_i = rng.integers(0, 3), rng.integers(0, 3)
            distance = rng.uniform(150, 1500)
            base_rpm = 2.0 + 0.1 * np.sin(2 * np.pi * date.dayofweek / 7)
            posted_rate = distance * base_rpm * rng.uniform(0.9, 1.1)
            rows.append({
                "load_id": f"SYN-{load_id:06d}",
                "pickup": pickups[pu_i], "delivery": deliveries[de_i],
                "pickup_lat": 38.0 + pu_i, "pickup_lon": -77.0 - pu_i,
                "delivery_lat": 39.0 + de_i, "delivery_lon": -78.0 - de_i,
                "distance": distance,
                "equipment": rng.choice(["Dry Van", "Reefer", "Flatbed"]),
                "weight": rng.uniform(10_000, 40_000),
                "date": date.strftime("%Y-%m-%d"),
                "posted_rate": posted_rate,
            })
    return pd.DataFrame(rows)


def test_fit_predict_smoke_test_returns_positive_finite_rates():
    train_df = _synthetic_shipment_frame(n_days=150)
    fitted = fit(train_df)

    future_dates = pd.date_range(train_df["date"].max(), periods=10, freq="D")[1:]  # next 9 days
    apply_df = pd.DataFrame({
        "load_id": [f"APP-{i:03d}" for i in range(len(future_dates))],
        "pickup": ["Richmond"] * len(future_dates),
        "delivery": ["Baltimore"] * len(future_dates),
        "pickup_lat": [38.0] * len(future_dates), "pickup_lon": [-77.0] * len(future_dates),
        "delivery_lat": [39.0] * len(future_dates), "delivery_lon": [-78.0] * len(future_dates),
        "distance": [400.0] * len(future_dates),
        "equipment": ["Dry Van"] * len(future_dates),
        "weight": [30_000.0] * len(future_dates),
        "date": [d.strftime("%Y-%m-%d") for d in future_dates],
    })

    predictions = predict(fitted, apply_df)
    assert len(predictions) == len(apply_df)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_predict_handles_a_frame_with_no_lat_lon_columns():
    """december-chart-inputs.csv has only pickup,delivery,distance,equipment,weight,date --
    no pickup_lat/pickup_lon/delivery_lat/delivery_lon columns at all. Every city has one
    fixed lat/lon across the training data, so predict() must look them up by name rather
    than require the caller to supply coordinates for a known, already-seen city.
    """
    train_df = _synthetic_shipment_frame(n_days=150)
    fitted = fit(train_df)

    future_date = (pd.to_datetime(train_df["date"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    apply_df = pd.DataFrame({
        "pickup": ["Richmond"],
        "delivery": ["Baltimore"],
        "distance": [400.0],
        "equipment": ["Dry Van"],
        "weight": [30_000.0],
        "date": [future_date],
    })
    assert "pickup_lat" not in apply_df.columns

    predictions = predict(fitted, apply_df)
    assert len(predictions) == 1
    assert np.isfinite(predictions[0])
    assert predictions[0] > 0


def test_predict_handles_a_novel_lane_not_seen_in_training():
    train_df = _synthetic_shipment_frame(n_days=150)
    fitted = fit(train_df)

    apply_df = pd.DataFrame({
        "load_id": ["NOVEL-001"],
        "pickup": ["Dallas"], "delivery": ["Reno"],  # combo never seen together in training
        "pickup_lat": [32.0], "pickup_lon": [-96.0],
        "delivery_lat": [39.0], "delivery_lon": [-119.0],
        "distance": [1200.0],
        "equipment": ["Reefer"],
        "weight": [25_000.0],
        "date": [(train_df["date"].max())],
    })
    apply_df["date"] = pd.to_datetime(apply_df["date"]) + pd.Timedelta(days=1)
    apply_df["date"] = apply_df["date"].dt.strftime("%Y-%m-%d")

    predictions = predict(fitted, apply_df)
    assert len(predictions) == 1
    assert np.isfinite(predictions[0])
    assert predictions[0] > 0
