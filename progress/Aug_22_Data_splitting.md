# Freight Rate Prediction — Aug 22nd: Data Splitting Strategy

Follow-up to [Aug_21st_Plan.md](Aug_21st_Plan.md) Phase 0. Question: given
[KNOWLEDGE.md](../KNOWLEDGE.md) §2 (train/dev/test splitting for time series), which scheme
actually fits this dataset, and what does the current `src/data/make_splits.py` still need?

## Recommendation

**Keep the fixed Sep–Oct holdout as the untouched final test set, and add expanding-window
(rolling-origin) CV with a 2-month horizon for model selection/tuning** — i.e. the KNOWLEDGE.md
§2.5 recipe, using expanding windows (§2.2) rather than sliding windows (§2.3).

## Why this shape

**The task's real evaluation is a single 2-month-ahead forecast.** Train on Jan–Oct, predict
Nov–Dec from one forecast origin (Oct 31). The internal evaluation should have the same shape:
train through some origin, predict the *next two months as one block*. The existing
`make_splits.py` (train < Sep 1, dev = Sep, holdout = Oct) already approximates this once
(origin Aug 31 → Sep is +1 month, Oct is +2 months), so it's the right skeleton — it just needs
company, not replacement.

**Why expanding-window CV on top, not just the single holdout.** A single fixed holdout gives
one evaluation window with high score variance and no uncertainty estimate (§2.1's stated
weakness). With 10 months of data we can afford 3 rolling origins at horizon h = 2 months:

| Fold | Train | Validate |
|---|---|---|
| 1 | Jan–Apr | May–Jun |
| 2 | Jan–Jun | Jul–Aug |
| 3 | Jan–Aug | Sep–Oct |

Tune hyperparameters and compare models on mean ± std across folds, weighting fold 3 most
heavily (§5.2: make the dev set resemble the test regime — Sep–Oct is the block adjacent to
Nov–Dec). Final ritual: retrain the chosen config on all of Jan–Oct, predict Nov–Dec once.

**Why expanding, not sliding (§2.3).** Sliding windows earn their keep when old regimes
actively mislead. The EDA found the opposite — a mild mid-year bump (Jan ~$2,256 → Jun ~$2,497
→ Oct ~$2,379), no evidence of a structural break — and with only 10 months of history,
discarding early months costs more than any staleness it removes. Revisit if the daily-index
diagnostics (ACF/PACF/ADF/KPSS, still open from the EDA review) turn up a regime shift.

**Why the gap/embargo (§2.4) mostly doesn't apply here.** The embargo exists so a training
row's lag/rolling features don't overlap the validation window. Per-load features here
(distance, lane, equipment, weight) have no temporal lookback, so this isn't the live risk.
The mirror-image discipline is what matters instead: **any time-dependent feature computed for
a validation row must be frozen at the fold's forecast origin, not computed from the row's own
date.** December inference won't have November's actual daily rates, so if a fold computes a
"lag-1 daily mean rate" feature for an October validation row using real September data, that
fold leaks and will flatter exactly the model that then fails on Nov–Dec. Every time-dependent
feature must be computed as of the origin (or produced by a recursive forecast), for train and
validation folds alike.

## Two caveats no split fixes

1. **No fold sees a full seasonal cycle.** There's no 2024 data, so every fold tests
   extrapolation of the observed Jan–Oct pattern, not genuine seasonal generalization — Nov–Dec
   holiday-season dynamics are unobserved in any fold. Argues for simple, robust trend handling
   over aggressive date features. Flag as irreducible risk in the report.
2. **Novel-lane rate should be checked per fold.** Validation has 17.6% unseen lanes; each CV
   fold should report its own unseen-lane rate against its own training window (fold 1's
   4-month window will have more novel lanes than fold 3's 8-month window), so fold scores are
   comparable and interpreted correctly.

## Concrete change to `src/data/`

`make_splits.py` stays as the final-holdout carver (Sep–Oct untouched until the very end,
optionally treated as one 2-month test block rather than separate dev/holdout months). Add a
`make_cv_folds(df, horizon="2M", n_folds=3)` alongside it for the three expanding-window
origins above. Tuning, early stopping, and model comparison run against the folds; Sep–Oct is
touched only once, by the finalist.
