# Freight Rate Prediction — Aug 25th: Results Summary (updated)

Consolidated writeup of the modeling work before moving on to the final Nov-Dec run against
`data/validation.csv` and `data/december-chart-inputs.csv`. Covers every original column's
processing, every engineered feature's formula, the modeling approach(es), and the validated
results. See [Aug_24_Plan.md](Aug_24_Plan.md) for the fuller day-by-day narrative this
summarizes.

**This version supersedes the same-named doc's first pass.** Two things changed the picture
materially after the first version was written: Pipeline A gained a `market_trend` feature
that closed most of its gap to Pipeline B, and swapping XGBoost in for LightGBM helped
Pipeline A further — see §4 for the full before/after. **Current recommendation: the
ensemble of Pipeline A + Pipeline B (`pipeline_c.ipynb`), MAE $105.02**, narrowly ahead of
Pipeline B alone ($105.79); see §5 for the tradeoff.

## 1. Original features and how each was processed

Source: `data/train-test.csv`, columns `load_id, pickup, delivery, pickup_lat, pickup_lon,
delivery_lat, delivery_lon, distance, equipment, weight, date, market_index, quote_signal,
posted_rate`.

| Column | Type | Processing | Used where? |
|---|---|---|---|
| `load_id` | id | none — identifier only, never a feature | nowhere |
| `pickup` | categorical (city, 64 levels) | not used directly as a raw categorical (too high-cardinality for one-hot, doesn't generalize to unseen cities on its own) — absorbed into `lane_target_enc` (§2) and into geometry via `pickup_lat`/`pickup_lon` | indirectly, both components |
| `delivery` | categorical (city, 64 levels) | same treatment as `pickup` | indirectly, both components |
| `pickup_lat`, `pickup_lon` | float | used raw, no scaling — both LightGBM and XGBoost are tree models, split on raw thresholds, invariant to monotonic rescaling of any one feature | raw, both components |
| `delivery_lat`, `delivery_lon` | float | used raw, no scaling (same reason) | raw, both components |
| `distance` | float (miles) | used raw, plus `log1p(distance)` engineered alongside it (§2) | raw + log, both components |
| `equipment` | categorical (3 levels: Dry Van, Flatbed, Reefer) | cast to pandas `category` dtype, native categorical handling in both LightGBM (`categorical_feature=[...]`) and XGBoost (`enable_categorical=True, tree_method="hist"`) — no one-hot, no scaling | both components |
| `weight` | float (lbs) | **data-quality fix**: some values are sign-flipped negative (a data-entry bug) — corrected with `weight = weight.abs()`. No further scaling | raw (post-fix), both components |
| `date` | date | not used directly as a model feature — source for every date/calendar engineered feature (§2), for Component A's `market_trend`, and for Component B's Stage 1 daily aggregation | indirectly, both components |
| `market_index` | float | **Component A**: kept, `use_market=True` (established by CV, `pipeline_a.ipynb` §3 — worth ~$10 MAE / ~5% relative on top of `market_trend`). Native NaN handling, no imputation. **Component B**: dropped entirely — its whole design point is not needing it (Stage 1 forecasts the market level directly from `date`), and it's absent from `december-chart-inputs.csv` | Component A only |
| `quote_signal` | float | same treatment as `market_index` — kept for Component A (~51 feature-importance splits, weak but real per `Aug_24_market_index.md`), dropped entirely for Component B | Component A only |
| `posted_rate` | float ($) | **target.** Component A: `log1p(posted_rate)`, back-transformed with `expm1`. Component B: re-expressed as `log(posted_rate / (daily_level(date) x distance))` — divides out both the dominant distance effect and the day-to-day market level before the tree model sees it; back-transformed with `exp(...) x daily_level x distance` | target, both components (different transforms) |

**No feature scaling/normalization (StandardScaler, min-max, etc.) is used anywhere.** Both
LightGBM and XGBoost (and the ETS/SARIMA/STL candidates, which operate on the raw daily
$/mile series) don't need it — trees split on raw thresholds and are invariant to monotonic
transforms of any one feature. The only numeric transforms applied are `log1p`/`log`, for
target-shape and signal-separation reasons (§2/§3), not scale-normalization.

## 2. Engineered features and formulas

All fit **OOF-safe** — any statistic derived from the target (the lane encoder, the market
trend/level forecasts) is fit on the current fold's/pool's *train* split only, then applied to
both train and validation/holdout, per KNOWLEDGE.md §1.3's leakage discipline.

**Shipment**
- `log_distance = log1p(distance)`

**Geography** (`haversine_miles` = great-circle distance, `R = 3958.8` mi):
```
haversine = 2R · arcsin( sqrt( sin²((lat2-lat1)/2) + cos(lat1)·cos(lat2)·sin²((lon2-lon1)/2) ) )
circuity  = distance / haversine        # how indirect the actual route is vs. straight-line
lon_delta = delivery_lon - pickup_lon   # crude east/west direction signal
```

**Lane target encoding** (`lane_target_enc`) — count-weighted empirical-Bayes shrinkage,
replacing an earlier discrete lane→city→global fallback chain (Aug 23-24 debug). Let
`y = log1p(posted_rate)` on the train split only:
```
city_prior_pickup   = (n_pickup   · mean_pickup   + k_city · global_mean) / (n_pickup   + k_city)
city_prior_delivery = (n_delivery · mean_delivery + k_city · global_mean) / (n_delivery + k_city)
city_prior           = (city_prior_pickup + city_prior_delivery) / 2

lane_target_enc = (n_lane · mean_lane + k_lane · city_prior) / (n_lane + k_lane)
```
`mean_*`/`n_*` are `y`'s group mean/count over `pickup`, `delivery`, or the `(pickup,
delivery)` lane pair. `k_lane=60, k_city=15` (CV grid search, `pipeline_a.ipynb` §3b). A novel
lane (`n_lane=0`) collapses exactly to `city_prior`. Validated on a leave-lane-out split
(17.5% of lanes held out entirely, 5 seeds): MAE $102.91 vs. $791.89 for the old fallback
chain. Shared by both components.

**Market-level trend** (`market_trend`, Component A only) — STL decomposition + damped Holt
forecast of the daily median $/mile series, replacing raw `market_index` as the primary
market-level signal (`Aug_24_market_index.md`):
```
daily(d)      = median({ posted_rate_i / distance_i : date_i = d })     # per calendar day, train split only
trend         = STL(daily, period=7, robust=True).trend                 # smooth component, seasonality stripped
market_trend  = trend(d)  if d already observed
              = DampedHolt(trend).forecast(d)  otherwise                # never an unbounded linear extrapolation
```
Exists for every date by construction (model output, not a table lookup) — zero missingness
on both `validation.csv`'s sparse `market_index` gaps and `december-chart-inputs.csv`'s total
absence of the column, with one mechanism. Outperforms raw `market_index` in feature
importance (164 vs. 98 splits) despite being the smoothed, lower-information version of the
same signal — the model leans on the denoised level harder than the noisy per-load read.

**Date / calendar** (`doy`=day-of-year, `dow`=day-of-week 0-6):
```
month_sin = sin(2π · month/12)     month_cos = cos(2π · month/12)
dow_sin   = sin(2π · dow/7)        dow_cos   = cos(2π · dow/7)
doy_sin   = sin(2π · doy/365)      doy_cos   = cos(2π · doy/365)
is_weekend        = 1 if dow ∈ {5,6} else 0
days_since_start  = (date - 2025-01-01).days                  # trend proxy
days_to_christmas = (2025-12-25 - date).days
days_to_new_year  = (2026-01-01 - date).days
```
No month×equipment interactions (EDA §13: seasonal shape is equipment-invariant). The 6
sin/cos columns were ablation-tested for redundancy given the shallow (`num_leaves=7`) tree
budget and found to matter (dropping them cost $9.58 MAE, `pipeline_a.ipynb` §4c) — LightGBM
this shallow behaves more like an additive/GAM-style booster than a flexible tree ensemble,
and needs the same periodic basis functions a linear model would.

**Component B's Stage 1** (daily market-level forecast, feeds Stage 2's target rather than
being a feature — see §3): ETS with additive damped trend + additive weekly seasonal, chosen
by backtesting 5 candidates (seasonal-naive, drift, ETS, SARIMA(1,1,1)(1,1,1)₇, and
`stl_trend` — the same STL+damped-Holt method as `market_trend` above, tried as a Stage 1
candidate and rejected, see §3).

## 3. Modeling approach

Two components, each independently developed, then blended.

### Component A — flat XGBoost + `market_trend`

One GBDT on `log1p(posted_rate)` directly, using calendar features plus `market_trend` as its
handle on "where the market is today." `distance`/`log_distance` monotone-constrained
(carry ~94% of the raw log-rate signal, EDA §10). **XGBoost**, not LightGBM (`pyproject.toml`
had `xgboost` pinned and installed the whole time, just never tried until Aug 25):
`learning_rate=0.02`, `max_depth=3` (level-wise growth; stands in for LightGBM's
`num_leaves=7` leaf budget — the two libraries' default growth strategies differ, leaf-wise
vs. level-wise, which is part of why the swap mattered), `subsample=0.8`,
`colsample_bytree=0.8`, early stopping at 300 rounds. Untranslated params
(`min_child_weight`, `reg_alpha/lambda`) left at XGBoost defaults rather than guess a
translation for LightGBM's `min_child_samples` — the two aren't equivalent (sample count vs.
Hessian sum).

### Component B — two-stage (ETS + LightGBM offset)

Splits "what does a load cost" from "where is the market today" into two explicit stages
instead of handing the market-level problem to one flat model's calendar features.

**Stage 1**: daily median $/mile forecast. ETS won the 5-candidate backtest at mean CV MAE
**$0.083/mi** (drift $0.088, SARIMA $0.089, seasonal-naive $0.091, `stl_trend` $0.111 — worst
of the five, see §2).

**Stage 2**: LightGBM on `log(posted_rate / (daily_level(date) x distance))`.
`learning_rate=0.02`, `num_leaves=7`, `min_child_samples=30`, `subsample=0.8`,
`colsample_bytree=0.8`, early stopping at 300 rounds. **No monotone constraint** — distance's
dominant effect is already divided out of this target.

**Combining stages at inference:**
```
predicted_rate_B = exp(stage_2_offset_pred(features)) x stage_1_forecast(date) x distance
```

### The ensemble (Pipeline C)

```
predicted_rate = w_a x predicted_rate_A + (1 - w_a) x predicted_rate_B
```
`w_a` grid-searched on the 3 CV folds (minimizing mean MAE), applied once to the holdout —
same discipline as every other hyperparameter in this project. Current selection: `w_a=0.75`.

**Validation harness (shared)**: `ExpandingWindowCVSplitter` (3 walk-forward folds, Jan-Aug
pool) for all model-selection decisions, plus an untouched Sep-Oct holdout scored exactly
once per configuration. `n_estimators` for every final fit picked via a dedicated
early-stopping tail carved from the pool (last 2 months) rather than reusing CV-fold iteration
counts (Aug 23 bug fix).

## 4. Results — how we got here

Sep-Oct holdout (touched once per configuration), dollar-space `posted_rate` metrics:

| Step | Model | MAE | RMSE | SMAPE | MASE |
|---|---|---|---|---|---|
| | naive (lane-median $/mi × distance) | $194.50 | $658.22 | 7.47% | 1.000 |
| | linear (log-dist + equipment, 3 feat) | $134.59 | $639.77 | 5.16% | 0.692 |
| Aug 23 | Pipeline A, tuned LightGBM, no encoder fix | $151.48 | $651.78 | 5.73% | 0.779 |
| Aug 24 | Pipeline A, + shrinkage lane encoder | $128.09 | $639.59 | 4.73% | 0.659 |
| Aug 24 | Pipeline B, two-stage (LightGBM Stage 2) | $105.79 | $632.51 | 3.85% | 0.544 |
| Aug 24 | Pipeline C, ensemble of the above two | $121.93 | $637.67 | 4.48% | 0.627 — **worse** than B alone |
| Aug 25 | Pipeline A, + `market_trend` feature | $109.93 | $633.45 | 3.98% | 0.565 |
| Aug 25 | Pipeline A, + XGBoost (same features) | $106.90 | $632.66 | 3.85% | 0.550 |
| Aug 25 | Pipeline B, XGBoost Stage 2 (no change) | $106.46 | $632.45 | 3.89% | 0.547 — not adopted, LightGBM stays |
| **Aug 25** | **Pipeline C, ensemble rebuilt (A=XGBoost+trend, B=LightGBM)** | **$105.02** | **$632.30** | **3.77%** | **0.540** |

**The A-vs-B gap closed from 17% to ~1%.** `market_trend` alone took Pipeline A from $128.09
to $109.93; the XGBoost swap took it to $106.90. Once the two architectures were close in
quality, the ensemble — which *hurt* at the original 17% gap (diagnosed as fold 2's Stage-1
bias skewing the CV weight search toward the weaker component) — now helps, landing at
$105.02, better than either component alone on every metric. Full mechanism write-up in
`pipeline_c.ipynb`'s summary and [Aug_24_Plan.md](Aug_24_Plan.md).

**Known, disclosed limitation** (not fixable by retuning): Stage 1's CV fold 2 (train Jan-Jun
→ val Jul-Aug) hit a genuine, unforeshadowed trend reversal — confirmed by plotting the
forecast against the actual daily series (`pipeline_b.ipynb` §3c), and a seasonality-free
alternative (`stl_trend`) was tried specifically to test whether a simpler method would be
more robust there and made it *worse*, not better. No history-only forecaster could have
anticipated a reversal with zero precedent in its own training window. The final holdout
doesn't hit this (its training pool runs past the reversal), but the same risk applies in
principle to the Nov-Dec forecast — nothing in the historical data can rule out an
unforeshadowed reversal in that window either. State this plainly in the report/loom.

## 5. Ensemble vs. Pipeline B alone — the remaining decision

| | MAE | Complexity |
|---|---|---|
| Pipeline B alone (LightGBM Stage 2) | $105.79 | One feature pipeline, one model, one forecast (ETS) |
| Ensemble (Pipeline C) | $105.02 (-0.7%) | Two feature pipelines, two models (XGBoost + LightGBM), one forecast (ETS), a blend weight to carry through everywhere |

The ensemble is the best number on the holdout, and both components are already
built/trained, so it's not expensive to produce — but the gain over Pipeline B alone is
modest (0.7%), and it roughly doubles what has to be reproduced correctly on
`validation.csv`/`december-chart-inputs.csv` (two feature-engineering passes instead of one,
two models to keep in sync, `market_trend` alongside Stage 1's forecast). Not yet decided
which ships for the final run — flagged here rather than picked unilaterally.

## 6. What's left before the final run

1. **Decide: ensemble or Pipeline B alone** (§5).
2. Fit the chosen model(s) on the **full** `train-test.csv` (Jan-Oct, no holdout carve-out
   needed anymore — model selection is done).
3. Run on all 12,000 rows of `data/validation.csv` → `validation_predictions.csv`
   (`load_id,predicted_rate`). Confirmed `data/validation.csv` **does** include
   `market_index`/`quote_signal` (only the December chart file lacks them) — relevant for
   Component A (uses them) if the ensemble ships, moot for Component B either way.
4. Run on `data/december-chart-inputs.csv` (31 fixed-route December rows) → fill
   `predicted_rate`, validate + chart via `score.py`. Component A's `market_trend` and
   Component B's Stage 1 forecast both handle this file's total absence of `market_index`/
   `quote_signal` by construction (neither needs the raw columns for December dates).
5. Promote the now-settled feature/model code (geo/date features, shrinkage lane encoder,
   `market_trend`, Stage 1/2 fitting) out of the notebooks and into `src/`, per the Aug 22
   "promote once an architecture wins" convention — not done yet, still notebook-inline and
   now duplicated across three notebooks.
