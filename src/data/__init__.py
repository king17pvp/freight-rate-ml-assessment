from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TRAIN_TEST_PATH = _REPO_ROOT / "data" / "train-test.csv"


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw training CSV. Does NOT parse `date` -- every caller parses it itself
    right after loading, so this keeps the same two-line contract the notebooks already use.
    """
    csv_path = Path(path) if path is not None else _DEFAULT_TRAIN_TEST_PATH
    return pd.read_csv(csv_path)


@dataclass
class Splits:
    train: pd.DataFrame
    dev: pd.DataFrame
    holdout: pd.DataFrame


def make_splits(raw: pd.DataFrame) -> Splits:
    """Fixed 3-tier split: train < Sep, dev = Sep, holdout = Oct (Aug_22_Data_splitting.md).
    Returns raw's own rows/dtypes unchanged, just filtered -- callers parse `date` themselves.
    """
    dates = pd.to_datetime(raw["date"])
    train = raw[dates < "2025-09-01"].copy()
    dev = raw[(dates >= "2025-09-01") & (dates < "2025-10-01")].copy()
    holdout = raw[dates >= "2025-10-01"].copy()
    return Splits(train=train, dev=dev, holdout=holdout)


@dataclass
class ExpandingWindowCVConfig:
    first_train_end: str = "2025-04-30"
    horizon_months: int = 2
    n_folds: int = 3
    holdout_start: str = "2025-09-01"


@dataclass
class Fold:
    index: int
    train: pd.DataFrame
    val: pd.DataFrame


@dataclass
class CVResult:
    folds: list[Fold]
    holdout: pd.DataFrame


class ExpandingWindowCVSplitter:
    """3 expanding-window (rolling-origin) folds at a fixed 2-month horizon, plus a fixed
    Sep-Oct holdout, per progress/Aug_22_Data_splitting.md. `df["date"]` must already be
    parsed to datetime -- this mirrors every notebook's usage (`splitter.split(clean)` is
    always called after `clean["date"] = pd.to_datetime(...)`).
    """

    def __init__(self, config: ExpandingWindowCVConfig):
        self.config = config

    def split(self, df: pd.DataFrame) -> CVResult:
        cfg = self.config
        dates = df["date"]
        folds: list[Fold] = []
        train_end = pd.Timestamp(cfg.first_train_end)
        for fold_index in range(1, cfg.n_folds + 1):
            val_start = train_end + pd.Timedelta(days=1)
            val_end = val_start + pd.DateOffset(months=cfg.horizon_months) - pd.Timedelta(days=1)
            train_mask = dates <= train_end
            val_mask = (dates >= val_start) & (dates <= val_end)
            folds.append(Fold(index=fold_index, train=df[train_mask].copy(), val=df[val_mask].copy()))
            train_end = val_end

        holdout = df[dates >= pd.Timestamp(cfg.holdout_start)].copy()
        return CVResult(folds=folds, holdout=holdout)
