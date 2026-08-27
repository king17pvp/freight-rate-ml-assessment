import pandas as pd

from data import ExpandingWindowCVConfig, ExpandingWindowCVSplitter, load_data, make_splits


def _synthetic_frame(start="2025-01-01", periods=304):
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame({"date": dates, "value": range(periods)})


def test_load_data_reads_train_test_csv_with_unparsed_date():
    df = load_data()
    assert len(df) == 48_000
    assert df["date"].dtype == object  # not parsed -- caller's job, matches notebook usage
    assert set(df.columns) >= {"load_id", "pickup", "delivery", "distance", "posted_rate", "date"}


def test_load_data_accepts_explicit_path(tmp_path):
    custom = tmp_path / "tiny.csv"
    pd.DataFrame({"date": ["2025-01-01"], "value": [1]}).to_csv(custom, index=False)
    df = load_data(custom)
    assert len(df) == 1


def test_make_splits_three_tiers_are_contiguous_and_disjoint():
    raw = _synthetic_frame()  # Jan 1 -> Oct 31, 2025 (304 days)
    splits = make_splits(raw)
    assert splits.train["date"].min() == pd.Timestamp("2025-01-01")
    assert splits.train["date"].max() == pd.Timestamp("2025-08-31")
    assert splits.dev["date"].min() == pd.Timestamp("2025-09-01")
    assert splits.dev["date"].max() == pd.Timestamp("2025-09-30")
    assert splits.holdout["date"].min() == pd.Timestamp("2025-10-01")
    assert splits.holdout["date"].max() == pd.Timestamp("2025-10-31")
    assert len(splits.train) + len(splits.dev) + len(splits.holdout) == len(raw)


def test_expanding_window_cv_fold_boundaries_match_spec():
    df = _synthetic_frame()
    df["date"] = pd.to_datetime(df["date"])
    splitter = ExpandingWindowCVSplitter(ExpandingWindowCVConfig())
    result = splitter.split(df)

    assert len(result.folds) == 3
    fold1, fold2, fold3 = result.folds
    assert (fold1.train["date"].min(), fold1.train["date"].max()) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-30"))
    assert (fold1.val["date"].min(), fold1.val["date"].max()) == (pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-30"))
    assert (fold2.train["date"].min(), fold2.train["date"].max()) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30"))
    assert (fold2.val["date"].min(), fold2.val["date"].max()) == (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-31"))
    assert (fold3.train["date"].min(), fold3.train["date"].max()) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-08-31"))
    assert (fold3.val["date"].min(), fold3.val["date"].max()) == (pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-31"))

    assert result.holdout["date"].min() == pd.Timestamp("2025-09-01")
    assert result.holdout["date"].max() == pd.Timestamp("2025-10-31")
    # fold 3's val and the holdout cover the same window by design (Aug_22_Data_splitting.md):
    # there's no data past October, so the last expanding-window fold necessarily previews
    # the fixed holdout window.
    pd.testing.assert_frame_equal(fold3.val.reset_index(drop=True), result.holdout.reset_index(drop=True))
