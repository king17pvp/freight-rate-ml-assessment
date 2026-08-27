from __future__ import annotations

import numpy as np
import pandas as pd

K_LANE, K_CITY = 60, 15  # selected by pipeline_a.ipynb's CV grid search -- not re-derived


def fit_lane_target_encoder_shrink(train_df: pd.DataFrame, target: pd.Series, k_lane: float, k_city: float) -> dict:
    """Count-weighted empirical-Bayes shrinkage lane encoder.

    A lane's raw mean is pulled toward a city prior in proportion to how few observations it
    has, and the city prior itself is pulled toward the global mean the same way. A novel
    lane (0 observations) collapses exactly to the city prior.
    """
    y = target.reindex(train_df.index)
    global_mean = y.mean()

    pickup_stats = y.groupby(train_df["pickup"]).agg(["mean", "count"])
    delivery_stats = y.groupby(train_df["delivery"]).agg(["mean", "count"])
    pickup_shrunk = (pickup_stats["count"] * pickup_stats["mean"] + k_city * global_mean) / (pickup_stats["count"] + k_city)
    delivery_shrunk = (delivery_stats["count"] * delivery_stats["mean"] + k_city * global_mean) / (delivery_stats["count"] + k_city)

    lane_stats = y.groupby([train_df["pickup"], train_df["delivery"]]).agg(["mean", "count"])
    return {
        "lane_mean": lane_stats["mean"],
        "lane_count": lane_stats["count"],
        "pickup_shrunk": pickup_shrunk,
        "delivery_shrunk": delivery_shrunk,
        "global": global_mean,
        "k_lane": k_lane,
    }


def apply_lane_target_encoder_shrink(df: pd.DataFrame, encoder: dict) -> pd.Series:
    global_mean, k_lane = encoder["global"], encoder["k_lane"]
    pu = encoder["pickup_shrunk"].reindex(df["pickup"]).to_numpy()
    de = encoder["delivery_shrunk"].reindex(df["delivery"]).to_numpy()
    pu = np.where(np.isnan(pu), global_mean, pu)
    de = np.where(np.isnan(de), global_mean, de)
    city_prior = (pu + de) / 2.0

    lane_idx = pd.MultiIndex.from_arrays([df["pickup"], df["delivery"]])
    lane_mean = np.nan_to_num(encoder["lane_mean"].reindex(lane_idx).to_numpy(), nan=0.0)
    lane_count = np.nan_to_num(encoder["lane_count"].reindex(lane_idx).to_numpy(), nan=0.0)

    enc = (lane_count * lane_mean + k_lane * city_prior) / (lane_count + k_lane)
    return pd.Series(enc, index=df.index, name="lane_target_enc")
