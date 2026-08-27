#!/usr/bin/env python
"""Fit Pipeline B on the full training pool and produce the two submission artifacts:
`validation_predictions.csv` (12,000 rows, load_id + predicted_rate) and a filled copy of
`data/december-chart-inputs.csv` (predicted_rate column populated, all other columns and
row order untouched -- score.py checks both column order and the fixed field values).

Usage:
    uv run python scripts/generate_submission.py
    uv run python scripts/generate_submission.py --train-test path/to/train-test.csv \
        --validation path/to/validation.csv --december path/to/december-chart-inputs.csv \
        --output-dir .
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_b import fit, predict

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-test", default=REPO_ROOT / "data" / "train-test.csv", type=Path)
    parser.add_argument("--validation", default=REPO_ROOT / "data" / "validation.csv", type=Path)
    parser.add_argument("--december", default=REPO_ROOT / "data" / "december-chart-inputs.csv", type=Path)
    parser.add_argument("--output-dir", default=REPO_ROOT, type=Path)
    args = parser.parse_args()

    print(f"loading training pool from {args.train_test} ...")
    train_df = pd.read_csv(args.train_test)

    print("fitting Pipeline B on the full pool (no holdout carve-out -- model selection is done) ...")
    fitted = fit(train_df)

    print(f"predicting {args.validation} ...")
    validation_df = pd.read_csv(args.validation)
    validation_preds = predict(fitted, validation_df)
    out_validation = pd.DataFrame({
        "load_id": validation_df["load_id"],
        "predicted_rate": validation_preds,
    })
    validation_out_path = args.output_dir / "validation_predictions.csv"
    out_validation.to_csv(validation_out_path, index=False)
    print(f"wrote {len(out_validation)} rows -> {validation_out_path}")

    print(f"predicting {args.december} ...")
    december_df = pd.read_csv(args.december)
    december_preds = predict(fitted, december_df.drop(columns=["predicted_rate"]))
    december_out = december_df.copy()
    december_out["predicted_rate"] = december_preds
    december_out_path = args.output_dir / "december-chart-inputs-filled.csv"
    december_out.to_csv(december_out_path, index=False)
    print(f"wrote {len(december_out)} rows -> {december_out_path}")

    print("\ndone. Run the scorer next:")
    print(
        f"  uv run python score.py --predictions {validation_out_path} "
        f"--december-predictions {december_out_path}"
    )


if __name__ == "__main__":
    main()
