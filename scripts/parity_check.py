#!/usr/bin/env python
"""One-time parity check: does src/two_stage_model.py reproduce pipeline_b.ipynb's holdout MAE?

Refits via the new src/ code on the same Jan-Aug pool -> Sep-Oct holdout split the notebook
used (ExpandingWindowCVSplitter's default holdout, cell 20 of pipeline_b.ipynb), and compares
against the notebook's recorded $105.79 MAE (progress/Aug_25_Results.md §4). Not exact-bit
reproduction (ETS's optimizer and LightGBM's training can differ in the last few floating
point digits run-to-run) -- looking for "close", not "identical". If this doesn't land within
a few dollars of $105.79, something was lost in translation; find it before trusting the CLI's
real-data run (scripts/generate_submission.py) or committing validation_predictions.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import ExpandingWindowCVConfig, ExpandingWindowCVSplitter, load_data
from eval import mae
from two_stage_model import fit, predict

NOTEBOOK_HOLDOUT_MAE = 105.79
TOLERANCE_DOLLARS = 5.0  # "within floating point tolerance" -- a few dollars, not bit-exact


def naive_lane_rpm_predict(train_raw: pd.DataFrame, apply_raw: pd.DataFrame) -> np.ndarray:
    rpm = train_raw["posted_rate"] / train_raw["distance"]
    lane_rpm = rpm.groupby([train_raw["pickup"], train_raw["delivery"]]).median()
    global_rpm = rpm.median()
    idx = pd.MultiIndex.from_arrays([apply_raw["pickup"], apply_raw["delivery"]])
    pred_rpm = lane_rpm.reindex(idx).to_numpy()
    pred_rpm = np.where(np.isnan(pred_rpm), global_rpm, pred_rpm)
    return pred_rpm * apply_raw["distance"].to_numpy()


def main() -> None:
    raw = load_data()
    raw["date"] = pd.to_datetime(raw["date"])
    clean = raw.copy()
    clean["weight"] = clean["weight"].abs()

    splitter = ExpandingWindowCVSplitter(ExpandingWindowCVConfig())
    result = splitter.split(clean)
    pool = clean[clean["date"] < "2025-09-01"]
    holdout = result.holdout

    print(f"pool: {len(pool)} rows ({pool['date'].min().date()} -> {pool['date'].max().date()})")
    print(f"holdout: {len(holdout)} rows ({holdout['date'].min().date()} -> {holdout['date'].max().date()})")

    print("fitting the two-stage rate model on the pool ...")
    fitted = fit(pool)
    predictions = predict(fitted, holdout)

    holdout_mae = mae(holdout["posted_rate"].to_numpy(), predictions)
    naive_predictions = naive_lane_rpm_predict(pool, holdout)
    naive_mae = mae(holdout["posted_rate"].to_numpy(), naive_predictions)

    print(f"\nsrc/two_stage_model.py holdout MAE: ${holdout_mae:.2f}")
    print(f"notebook's recorded holdout MAE: ${NOTEBOOK_HOLDOUT_MAE:.2f}")
    print(f"naive baseline MAE (sanity check, should be much worse): ${naive_mae:.2f}")

    delta = abs(holdout_mae - NOTEBOOK_HOLDOUT_MAE)
    if delta <= TOLERANCE_DOLLARS:
        print(f"\nPASS -- within ${TOLERANCE_DOLLARS:.2f} tolerance (delta ${delta:.2f}).")
    else:
        print(f"\nFAIL -- delta ${delta:.2f} exceeds ${TOLERANCE_DOLLARS:.2f} tolerance.")
        print("Something was lost in translation from the notebook -- do not trust the CLI run yet.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
