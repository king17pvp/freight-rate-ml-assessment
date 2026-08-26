# Freight Rate Prediction — Aug 22nd: Pipeline Architecture Plan

Follow-up to [Aug_21st_Plan.md](Aug_21st_Plan.md) Phase 3–4, [Aug_22_Data_splitting.md](Aug_22_Data_splitting.md),
and [Aug_22_EDA_Addition.md](Aug_22_EDA_Addition.md). Question: given EDA §9 (the daily
market level is unit-root-like, non-stationary, with weekly seasonality) and §15
(`market_index`'s daily mean tracks that level at r=0.82), should the model learn "what a
load costs" and "where the market is today" jointly, or as two explicit stages?

## The tension

Two distinct signals live in this dataset:

1. **What a load costs given its own attributes** — distance, equipment, lane. Stable,
   well-behaved: distance+equipment alone already explains R²=0.942 of log-rate (EDA §10).
2. **Where the overall market is on a given day** — this drifts (ADF/KPSS both call the
   daily level non-stationary, STL trend strength 0.91) and is exactly what must be
   *forecast* into Nov–Dec, a range with no observed seasonal cycle to learn from.

## Two candidate architectures

### Pipeline A — Flat (single model)

One GBDT on `log(posted_rate) ~ distance + equipment + lane/geo features + date features + ...`.
Date features (month, day-of-year, cyclical encodings, days-to-holiday) are the model's only
handle on "where the market is today" — it must infer the daily level implicitly from
calendar position.

- **Pros**: simple, one model to tune/explain, trees capture nonlinear interactions
  automatically, fastest to ship end-to-end.
- **Cons**: trees cannot extrapolate (KNOWLEDGE.md §3.2) — Nov–Dec calendar values are
  entirely outside the training range, so a tree has never split on "day of year 330" and
  falls back to the nearest seen leaf. Asks one model to both price a load *and* forecast a
  drifting market level from calendar features alone.

### Pipeline B — Two-stage

**Stage 1**: forecast the daily market index (daily median $/mile, or the daily mean of
`market_index`) into Nov–Dec — a genuine small time-series problem (ARIMA/ETS/trend+weekly-
seasonal per KNOWLEDGE.md §3.1, not a big tabular one).

**Stage 2**: train a model on `log(posted_rate / daily_index) ~ distance + equipment + lane
+ ...` — predict each load's *offset* from that day's market level, a target with the drift
removed. At inference: `predicted_rate = stage_1_forecast(date) × stage_2_offset(features)`.

- **Pros**: separates the extrapolation problem (needs a trend-aware method) from the
  cross-sectional pricing problem (trees excel at this); directly addresses the
  non-stationarity finding instead of hoping calendar features paper over it; `market_index`
  becomes a genuinely useful training-time proxy for stage 1 even though it's absent at
  Nov–Dec inference (stage 1 forecasts it forward, so the row-level model never needs it).
- **Cons**: two models to build/tune/validate; stage 1's forecast error propagates
  multiplicatively into every stage-2 prediction; more code; harder to explain if time-boxed.

## Decision: build both, in order, and compare

1. **Ship Pipeline A first** (`notebooks/pipeline_a.ipynb`) — flat GBDT + strong date
   features. Fastest path to an end-to-end, gradeable pipeline; establishes the working
   baseline against the EDA §16 floors (MAE $135 / RMSE $632, linear log-rate ~
   log-distance + equipment, fit Jan–Jun → eval Jul–Aug).
2. **Then build Pipeline B** (`notebooks/pipeline_b.ipynb`) as the explicit comparison,
   demonstrating the extrapolation-risk diagnosis is more than theoretical.
3. **Compare on the `ExpandingWindowCVSplitter` folds** (train/data/make_splits.py) — same
   3 folds for both pipelines, same untouched Sep–Oct holdout touched once by the finalist.
   Metrics: MAE/RMSE (back-transformed `expm1`) + SMAPE (`src/eval/metrics.py`), plus MASE
   against the naive seasonal baseline per KNOWLEDGE.md §3.
4. Whichever pipeline wins on the folds (or an ensemble of both, per KNOWLEDGE.md §5.1 —
   "ensembling is nearly free accuracy") becomes the model retrained on train+dev for the
   final December/validation predictions.

## Pipeline A — experiment design (this session's next step)

All work stays in `notebooks/pipeline_a.ipynb` for now — no changes to `src/`, so feature
logic and cleaning are inline in notebook cells, not extracted into `src/features/` or
`src/data/clean.py` yet. Promote to real modules once an architecture wins.

**Cleaning** (inline, per EDA findings):
- `weight = weight.abs()` (sign-flip bug, EDA data-quality note)
- `market_missing` flag; leave `market_index`/`weight` NaN for native GBDT handling

**Features** (inline):
- Shipment: `distance`, `log1p(distance)`, `weight`, `equipment` (categorical)
- Geography: `pickup_lat/lon`, `delivery_lat/lon`, circuity ratio (EDA §11), OOF lane target
  encoding with fallback chain (lane mean → pickup-city mean → delivery-city mean → global
  mean) to handle novel lanes (17.6% of validation)
- Date: month, day-of-week, day-of-year, cyclical (sin/cos) encodings, days-to-holiday,
  linear day-count-since-start (trend proxy) — no month×equipment interactions (EDA §13
  showed the seasonal shape is equipment-invariant)
- Market: `market_index`, `quote_signal` raw (kept as an in-sample exploratory feature for
  Pipeline A even though absent from `december-chart-inputs.csv` — matters for whether
  Pipeline A can even be used for the December chart, see "open question" below)

**Target**: `log1p(posted_rate)`, back-transform with `expm1`.

**Model**: LightGBM regressor, `objective="regression"` on log target, native categorical +
NaN handling, early stopping against each fold's validation window.

**Validation**: `ExpandingWindowCVSplitter` (3 folds, Jan→Aug pool) for model selection;
Sep–Oct holdout evaluated once at the end for the honest score.

**Diagnostics**: residuals by month, by equipment, by distance bucket, by novel-vs-seen lane
(expect worse error on novel lanes — quantify it), feature importances.

**Open question Pipeline A must answer**: since `market_index`/`quote_signal` aren't in
`december-chart-inputs.csv`, does including them help enough on the CV folds to justify a
fallback strategy at December-inference time (e.g. median fill), or should they be dropped
so Pipeline A's feature set is inference-safe by construction? Decide empirically — train
both a with- and without-market-features variant and compare fold scores.

## Pipeline A results (`notebooks/pipeline_a.ipynb`, run Aug 22)

**CV**: market features help on mean CV MAE ($210.99 vs $213.03 without) but are
inconsistent — they hurt the thinnest-history fold (fold 0, 2 months training) and help
folds 1-2. Selected `use_market=True` for the final model.

**Holdout (Sep-Oct, touched once)**: MAE $165.82 / RMSE $658.99 / SMAPE 6.06% / MASE 0.853
— beats the lane-median naive baseline by ~15%.

**The real finding — apples-to-apples baseline check** (same Jan-Aug pool -> Sep-Oct
holdout as the LightGBM run, not EDA §16's different Jul-Aug window):

| Model | MAE | RMSE | SMAPE | MASE |
|---|---|---|---|---|
| lane-median $/mi x distance (naive) | $194.50 | $658.22 | 7.47% | 1.000 |
| **linear log-rate ~ log-dist + equipment (3 features)** | **$134.59** | **$639.77** | **5.16%** | **0.692** |
| Pipeline A (LightGBM, 20-24 features) | $165.82 | $658.99 | 6.06% | 0.853 |

**A plain 3-feature linear model beat the tuned LightGBM.** Diagnosis: the model is
underfit, not fundamentally worse — mean CV best iteration was only ~72 trees at
`learning_rate=0.05` (early-stopping patience 100 cutting almost immediately on folds as
small as 9,255 rows), against a target (distance+equipment, R²=0.942 per EDA §10) that's
smooth and near-linear — exactly the shape trees are inefficient at approximating with few,
shallow trees. `lane_target_enc` and `equipment` dominate feature importance; the date/
market features barely got used.

**Second finding — novel-lane fragility, more urgent than flat-vs-two-stage**: the 19
novel-lane rows in the holdout (0.2% — Sep-Oct draws almost entirely from already-seen
lanes) have MAE $1,019.47 vs $163.93 for seen lanes — a 6.2x error multiplier. Real
validation is 17.6% novel lanes; if that multiplier holds anywhere close, novel-lane error
dominates the overall score. `lat/lon` features currently carry near-zero importance
(ranked below `pickup_lon` at 7 splits) — they are not doing the generalization job they
were included for.

## Underfitting bug — debugged and fixed (Aug 22, later same day)

Two issues were found in the final-model fit, one structural and one a hyperparameter
problem, plus a new finding that reframes both "next steps" above.

1. **Structural bug**: `n_estimators` for the final full-pool fit was set to the *mean* of
   the 3 CV folds' early-stopping iterations — but fold 0 trains on only 9,255 rows and
   stops almost immediately, dragging the mean down even though the final model trains on
   the full 38k-row pool and can support more rounds. **Fix**: carve a dedicated
   early-stopping tail from the pool itself (Jan-Jun fit -> Jul-Aug early-stop, mirroring
   EDA §16's window) instead of reusing CV-fold iteration counts. **Impact was small on its
   own** (best_iteration ~76, barely moved) — this bug was real but not the dominant cause.
2. **Real cause**: `learning_rate=0.05`/`num_leaves=31`/`patience=100` converged too fast for
   a near-linear relationship (distance+equipment, R²=0.942). Tuning to `learning_rate=0.02`,
   `num_leaves=7`, `patience=300`, plus a monotonic constraint on `distance`/`log_distance`,
   took the full-feature holdout MAE from **$165.82 -> $151.48** (RMSE $658.99 -> $651.78).
3. **New finding (ablation, notebook §4b)**: even tuned, the full-feature model still trails
   the 3-feature linear baseline ($151.48 vs $134.59). Refitting the same tuned LightGBM on
   *only* `distance`+`log_distance`+`equipment` gets MAE **$134.72** — matching the linear
   model almost exactly. **This proves the remaining gap is not underfitting or a GBDT
   weakness — it's that the extra 17-21 features are net-harmful.** `lane_target_enc` now
   dominates feature importance (1,043 splits vs. 421 for `distance`), consistent with a raw,
   unshrunk per-lane mean-log-rate encoding overfitting lanes with few observations.
4. **This unifies "retune" and "novel-lane fragility" into one root cause.** Novel-lane MAE
   is now $1,026.75 vs $149.54 seen (~6.9x, even starker than before) — the same
   `lane_target_enc` the model leans on hardest has the weakest fallback for unseen lanes.

## Revised next steps (before Pipeline B)

1. **Regularize `lane_target_enc`** — Bayesian/count-based shrinkage toward the
   pickup/delivery-city mean or global mean (instead of a raw group mean with a hard
   fallback chain), or replace the city-level fallback tier with a coarser regional encoding
   (k-means clusters on pickup/delivery lat/lon, per the original plan). This is now the
   single highest-leverage fix — likely to close both the linear-baseline gap on seen lanes
   and the novel-lane collapse at once, rather than being two separate problems.
2. **Build a leave-lane/city-out CV split** to validate the encoder fix against: the natural
   Sep-Oct holdout has only 19 novel-lane rows (0.2%), too few to trust the 6.2-6.9x
   multiplier or tune against reliably. Hold out ~15-20% of lanes/cities from the pool
   directly.
3. Once the encoder fix is validated on that larger novel-lane sample and Pipeline A
   reliably matches-or-beats the linear baseline on seen lanes, build Pipeline B and run the
   same CV-fold + holdout comparison.
