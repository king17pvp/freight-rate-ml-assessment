from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return 2 * r * np.arcsin(
        np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    )


def add_shipment_geo_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_distance"] = np.log1p(out["distance"])
    out["haversine"] = haversine_miles(out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"])
    out["circuity"] = out["distance"] / out["haversine"]
    out["lon_delta"] = out["delivery_lon"] - out["pickup_lon"]

    doy = out["date"].dt.dayofyear
    dow = out["date"].dt.dayofweek
    out["month_sin"] = np.sin(2 * np.pi * out["date"].dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["date"].dt.month / 12)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    out["is_weekend"] = (dow >= 5).astype(int)
    out["days_since_start"] = (out["date"] - pd.Timestamp("2025-01-01")).dt.days

    christmas = pd.Timestamp("2025-12-25")
    new_year = pd.Timestamp("2026-01-01")
    out["days_to_christmas"] = (christmas - out["date"]).dt.days
    out["days_to_new_year"] = (new_year - out["date"]).dt.days

    out["equipment"] = out["equipment"].astype("category")
    return out
