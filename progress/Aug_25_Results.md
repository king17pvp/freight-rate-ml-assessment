# Freight Rate Prediction — Aug 25th: Pipeline B Results Summary

Consolidated writeup of the chosen model — Pipeline B, the two-stage architecture
(`notebooks/pipeline_b.ipynb`) — before moving on to the final Nov-Dec run against
`data/validation.csv` and `data/december-chart-inputs.csv`. Covers every original column's
processing, every engineered feature's formula, the modeling approach, and the validated
results. See [Aug_24_Plan.md](Aug_24_Plan.md) for the full narrative (Pipelines A/B/C built
and compared) this summarizes.

## 1. Original features and how each was processed

Source: `data/train-test.csv`, columns `load_id, pickup, delivery, pickup_lat, pickup_lon,
delivery_lat, delivery_lon, distance, equipment, weight, date, market_index, quote_signal,
posted_rate`.

| Column | Type | Processing | Used in Pipeline B? |
|---|---|---|---|
| `load_id` | id | none — identifier only, never a feature | no |
| `pickup` | categorical (city, 64 levels) | not used directly as a raw categorical (too high-cardinality for one-hot, and doesn't generalize to unseen cities on its own) — absorbed into the `lane_target_enc` engineered feature (§2) and into geometry via `pickup_lat`/`pickup_lon` | indirectly (via `lane_target_enc`, `haversine`, `circuity`, `lon_delta`) |
| `delivery` | categorical (city, 64 levels) | same treatment as `pickup` | indirectly |
| `pickup_lat`, `pickup_lon` | float | used raw, no scaling — LightGBM is a tree model, splits on raw thresholds, invariant to any monotonic rescaling of an individual feature | yes, raw |
| `delivery_lat`, `delivery_lon` | float | used raw, no scaling (same reason) | yes, raw |
| `distance` | float (miles) | used raw, plus `log1p(distance)` engineered alongside it (§2) | yes, raw + log |
| `equipment` | categorical (3 levels: Dry Van, Flatbed, Reefer) | cast to pandas `category` dtype, passed to LightGBM's native categorical handling (`categorical_feature=[...]`) — no one-hot encoding, no scaling | yes |
| `weight` | float (lbs) | **data-quality fix**: some values are sign-flipped negative (a data-entry bug, not a real physical quantity) — corrected with `weight = weight.abs()`. No further scaling. | yes, raw (post-fix) |
| `date` | date | not used directly as a model feature — instead the source for every date/calendar engineered feature (§2) and for Stage 1's daily aggregation | indirectly (via engineered date features + Stage 1) |
| `market_index` | float | **dropped entirely.** EDA found it's a noisy *load-level* signal (many distinct values per calendar day), not a daily series — only its daily *mean* tracks the real daily price level (r=0.82 with daily median $/mile). Stage 1 forecasts that daily level directly from `date`, so `market_index` is never needed as a model input, and it's absent from `december-chart-inputs.csv` anyway | no |
| `quote_signal` | float | **dropped entirely**, same reasoning as `market_index` (also absent from `december-chart-inputs.csv`) | no |
| `posted_rate` | float ($) | **target.** Not scaled/normalized in the usual sense — instead re-expressed as the two-stage decomposition target (§3): `log(posted_rate / (daily_level(date) x distance))`. This divides out both the dominant distance effect and the day-to-day market level before the tree model ever sees it, then predictions are transformed back to dollars for every reported metric (`exp(...) x daily_level x distance`) | target |

**No feature scaling/normalization (StandardScaler, min-max, etc.) is used anywhere.**
LightGBM (and the ETS/SARIMA candidates, which operate on the raw daily $/mile series) don't
need it — trees split on raw thresholds and are invariant to monotonic transforms of any one
feature. The only numeric transform applied is `log1p`/`log`, and that's for target-shape and
signal-separation reasons (§2/§3), not for scale-normalization.

## 2. Engineered features and formulas

All fit **OOF-safe** — any statistic derived from the target (the lane encoder) is fit on the
current fold's/pool's *train* split only, then applied to both train and validation/holdout,
per KNOWLEDGE.md §1.3's leakage discipline.

**Shipment**
- `log_distance = log1p(distance)`

**Geography** (`haversine_miles` = great-circle distance, `R = 3958.8` mi):
```
haversine = 2R · arcsin( sqrt( sin²((lat2-lat1)/2) + cos(lat1)·cos(lat2)·sin²((lon2-lon1)/2) ) )
circuity  = distance / haversine        # how indirect the actual route is vs. straight-line
lon_delta = delivery_lon - pickup_lon   # crude east/west direction signal
```

**Lane target encoding** (`lane_target_enc`) — count-weighted empirical-Bayes shrinkage,
replacing an earlier discrete lane→city→global fallback chain (Aug 23-24 debug, see
[Aug_24_Plan.md](Aug_24_Plan.md)). Let `y = log1p(posted_rate)` on the train split only:
```
city_prior_pickup   = (n_pickup   · mean_pickup   + k_city · global_mean) / (n_pickup   + k_city)
city_prior_delivery = (n_delivery · mean_delivery + k_city · global_mean) / (n_delivery + k_city)
city_prior           = (city_prior_pickup + city_prior_delivery) / 2

lane_target_enc = (n_lane · mean_lane + k_lane · city_prior) / (n_lane + k_lane)
```
`mean_*`/`n_*` are `y`'s group mean/count over `pickup`, `delivery`, or the `(pickup,
delivery)` lane pair. `k_lane=60, k_city=15` (selected by CV grid search over
`k_lane∈{5,15,30,60}, k_city∈{5,15,30}`, `pipeline_a.ipynb` §3b). A novel lane (`n_lane=0`)
collapses exactly to `city_prior`; every lane in between is damped toward it by how little
history it has, instead of a hard-cutoff fallback. Validated on a dedicated leave-lane-out
split (17.5% of lanes held out entirely, 5 seeds): MAE $102.91 vs. $791.89 for the old
discrete fallback chain on the same split.

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
No month×equipment interaction terms — EDA §13 found the seasonal shape is
equipment-invariant, so they weren't worth the added complexity.

**Full Stage 2 feature list** (19 numeric + 1 categorical, no market features):
`distance, log_distance, weight, pickup_lat, pickup_lon, delivery_lat, delivery_lon,
circuity, lon_delta, month_sin, month_cos, dow_sin, dow_cos, doy_sin, doy_cos, is_weekend,
days_since_start, days_to_christmas, days_to_new_year, lane_target_enc, equipment`.

## 3. Modeling approach — Pipeline B (two-stage)

Splits "what does a load cost" from "where is the market today" into two explicit stages,
instead of asking one flat model to infer both from calendar features alone (Pipeline A's
approach — see §5 for that comparison).

**Stage 1 — daily market-level forecast.** Target: daily median $/mile
(`median(posted_rate/distance)` per calendar day, `asfreq("D")`, gap-interpolated). Backtested
4 candidates on 3 expanding-window CV folds and picked by MAE (no method assumed a priori):

| Candidate | Mean CV MAE ($/mi) |
|---|---|
| **ETS** (Holt-Winters, additive damped trend + additive weekly seasonal, period 7) | **0.083** — selected |
| drift | 0.088 |
| SARIMA(1,1,1)(1,1,1)₇ | 0.089 |
| seasonal-naive (`y_t = y_{t-7}`) | 0.091 |

**Stage 2 — cross-sectional LightGBM on the offset.** Target:
```
y = log( posted_rate / (daily_level(date) x distance) )
```
LightGBM regressor: `objective="regression"` (RMSE), `learning_rate=0.02`, `num_leaves=7`,
`min_child_samples=30`, `subsample=0.8`, `colsample_bytree=0.8`, up to `n_estimators=3000`
with early stopping (patience 300, temporal validation fold). **No monotone constraint** on
`distance` (unlike Pipeline A) — distance's dominant near-linear effect is already divided
out of this target, so forcing monotonicity is no longer justified.

**Combining stages at inference:**
```
predicted_rate = exp(stage_2_offset_pred(features)) x stage_1_forecast(date) x distance
```

**Validation harness**: same `ExpandingWindowCVSplitter` (3 walk-forward folds, Jan-Aug pool)
as every other pipeline in this project, plus an untouched Sep-Oct holdout scored exactly
once. `n_estimators` for the final Stage 2 fit picked via a dedicated early-stopping tail
carved from the pool (last 2 months) rather than reusing CV-fold iteration counts (Aug 23 bug
fix, applies to both Pipeline A and B).

## 4. Results

Sep-Oct holdout (touched once), dollar-space `posted_rate` metrics (`expm1`/`exp`
back-transformed, never scored in log space):

| Model | MAE | RMSE | SMAPE | MASE |
|---|---|---|---|---|
| naive (lane-median $/mi × distance) | $194.50 | $658.22 | 7.47% | 1.000 |
| linear (log-dist + equipment, 3 feat) | $134.59 | $639.77 | 5.16% | 0.692 |
| Pipeline A (flat LightGBM, shrinkage encoder) | $128.09 | $639.59 | 4.73% | 0.659 |
| **Pipeline B (two-stage, chosen)** | **$105.79** | **$632.51** | **3.85%** | **0.544** |
| Pipeline C (CV-selected ensemble of A+B) | $121.93 | $637.67 | 4.48% | 0.627 — worse than B alone, not used (`pipeline_c.ipynb`) |

**Pipeline B is the model going into the final run.** 46% MAE reduction vs. naive, 21% vs.
linear, ~17% vs. Pipeline A alone; MASE 0.544 means roughly half the error of the naive
baseline on the untouched holdout.

**Known, disclosed limitation** (not fixable by retuning — see [Aug_24_Plan.md](Aug_24_Plan.md)
Part 2b): Stage 1's CV fold 2 (train Jan-Jun → val Jul-Aug) hit a genuine trend reversal with
no precedent in its own training window — no history-only forecaster could have anticipated
it. The final holdout doesn't hit this (its training pool runs past the reversal), but the
same risk applies in principle to the Nov-Dec forecast: nothing in the historical data can
rule out an unforeshadowed reversal in that window either. Worth stating plainly in the
report/loom rather than hiding it — it's a structural limit of the method, not a bug.

## 5. What's left before the final run

1. Fit Pipeline B on the **full** `train-test.csv` (Jan-Oct, no holdout carve-out needed
   anymore — model selection is done).
2. Run it on all 12,000 rows of `data/validation.csv` → `validation_predictions.csv`
   (`load_id,predicted_rate`). Confirmed `data/validation.csv` **does** include
   `market_index`/`quote_signal` (only the December chart file lacks them) — moot for
   Pipeline B either way, since it uses neither.
3. Run it on `data/december-chart-inputs.csv` (31 fixed-route December rows) → fill
   `predicted_rate`, validate + chart via `score.py`.
4. Promote the now-settled feature/model code (geo/date features, shrinkage lane encoder,
   Stage 1/2 fitting) out of the notebook and into `src/`, per the Aug 22 "promote once an
   architecture wins" convention — not done yet, still notebook-inline.
