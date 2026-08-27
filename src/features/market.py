from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def build_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Daily median $/mile, reindexed to a full daily calendar with gaps interpolated."""
    rate_per_mile = df["posted_rate"] / df["distance"]
    daily = (
        df.assign(rate_per_mile=rate_per_mile)
        .groupby("date")
        .agg(rpm_median=("rate_per_mile", "median"), n=("rate_per_mile", "size"))
        .asfreq("D")
    )
    daily["rpm_median"] = daily["rpm_median"].interpolate(limit_direction="both")
    return daily


def ets_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Additive damped-trend, additive weekly-seasonal ETS -- won the 5-candidate Stage 1
    backtest (mean CV MAE $0.083/mi vs. $0.088-$0.111 for the alternatives,
    progress/Aug_25_Results.md §3).
    """
    model = ExponentialSmoothing(
        history, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
        initialization_method="estimated",
    ).fit()
    return model.forecast(horizon).to_numpy()
