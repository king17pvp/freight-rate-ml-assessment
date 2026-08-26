# Freight Rate Prediction — Aug 21st Plan

Goal: maximize prediction performance on `posted_rate` for the 12,000 validation loads
(Nov–Dec 2025) and the fixed December lane chart, following a conventional
EDA → cleaning → feature engineering → model development pipeline.

**Key data facts (from initial inspection):**

- Train: 48,000 rows, 2025-01-01 → 2025-10-31. Validation: 12,000 rows, **Nov–Dec 2025 (future)**.
- `distance` correlates 0.909 with `posted_rate` — dominant signal.
- 64 pickup + 64 delivery cities; 4,014 train lanes; **736 validation lanes never seen in train** → must generalize via city-level features, not lane memorization.
- Target right-skewed (median 2,031 / mean 2,374 / max 25,533) → **train on log(rate)**.
- NAs only in `weight` (0.6–1.4%) and `market_index` (0.8–2.1%) → native GBDT NaN handling is enough.
- `market_index` / `quote_signal` present in validation **but absent in December chart inputs** → model must cope with missing market features at inference (median fill), and the chart's signal must come from date features.

---

## Phase 0 — Project skeleton & validation strategy (the "custom test set")

The single most important design decision: **validation.csv is 2 months in the future**,
so any random split leaks time. We build our own test set the same way the real one is built.

### Split scheme (3 tiers)

| Tier | Definition | Purpose |
|---|---|---|
| **Train** | Jan 1 – Aug 31 (~38k rows) | Fit models |
| **Dev val** | Sep 1 – Sep 30 (~4.8k rows) | Iterate: feature/model selection, tuning |
| **Holdout test** | Oct 1 – Oct 31 (~4.9k rows) | Final, *touched as rarely as possible* — reports the honest generalization estimate |

Rules:

- The holdout is **time-contiguous and later than dev val** — it mimics the real train→validation gap.
- Every experiment reports dev-val RMSE first; only the finalist touches the holdout.
- Optional sanity check: expanding-window `TimeSeriesSplit` on Jan–Sep for stability of the choice.
- Metrics: RMSE and MAE on back-transformed rates (`expm1`), plus SMAPE (scale-free, defensible in the report). Optimize on log-scale RMSE (≈ relative error).

### Deliverables

- `src/data/` — loading + split logic (`make_splits.py`): deterministic seed, returns (X_train, X_dev, X_test).
- Baseline numbers recorded in this file as we go (see Phase 4).

---

## Phase 1 — EDA (`notebooks/01_eda.ipynb` or `src/eda/`)

Answers the "how does this market behave" questions that shape features.

1. **Target**: distribution (histogram, skew), log-transform justification, monthly trend (already known: Jan ~2,256 → Jun ~2,497 → Oct ~2,379), by-equipment medians (Reefer > Flatbed > Dry Van).
2. **Distance**: distribution, rate vs distance relationship (linear? breakpoints? empty miles), outliers (26k max rate — check).
3. **Geography**: city frequency, lane frequency, which cities are high/low rate (hubs vs rural), novel-lane structure (are they rare cities or new combos?).
4. **Date**: day-of-week effects, month seasonality, holiday proximity (Christmas/Thanksgiving), weekend effects.
5. **Market features**: `market_index` / `quote_signal` vs rate (scatter), their missingness pattern, whether they encode month/lane info.
6. **NAs & quality**: weight/market_index missing rows — pattern vs date/lane? Duplicated rows? implausible lat/lon (e.g., crossing state lines, zero-distance lanes)?
7. **Correlations**: pairwise, and conditional on equipment.

**Definition of done**: one page of key findings written into the report draft
(also feeds the Loom talking points).

---

## Phase 2 — Data cleaning

Minimal — the data is mostly clean. Do:

1. **NAs**: leave `weight`/`market_index` missing for GBDT (treat as informative). Document this choice.
2. **Outliers**: winsorize or clip extreme `posted_rate` (e.g., > 99.5th percentile) *only* if EDA shows they distort learning; prefer keeping and relying on log transform first.
3. **Dedup**: check `load_id` uniqueness; drop exact duplicate rows if any.
4. **Lat/lon sanity**: verify coordinates fall in the continental US; drop/fix impossible rows.
5. **Categorical hygiene**: strip whitespace/case from `pickup`, `delivery`, `equipment`; fix typos in city names (e.g., alternate spellings) so cardinalities stay at 64.

**Definition of done**: a single `clean()` function in `src/data/` that takes raw CSV → clean DataFrame, deterministic and idempotent.

---

## Phase 3 — Feature engineering (`src/features/`)

All features built in one `build_features()` that trains on train fold only (target encodings must be out-of-fold).

| Group | Features |
|---|---|
| Shipment | `distance`, `weight`, `equipment` (categorical), `distance*weight` interaction, `distance^2` (if EDA shows curvature) |
| Geography | `pickup`/`delivery` city (categorical), `pickup_lat/lon`, `delivery_lat/lon` (numerical — lets trees find regional neighborhoods, the only thing that generalizes to the 736 novel lanes), `lane_id` (pickup+delivery string) |
| Lane encodings | OOF target mean of `posted_rate` (log) per lane; fallback chain: lane mean → pickup-city mean → delivery-city mean → global mean (handles novel lanes) |
| Date | `month`, `day_of_week`, `week_of_year`, `day_of_year`, `days_to_christmas`, `days_to_new_year`, `is_weekend`, day-count-since-start (trend) |
| Market | `market_index`, `quote_signal` raw; flag `market_missing`; keep NA for December (or median-fill at inference) |
| Target | `log1p(posted_rate)` as y (back-transform with `expm1`) |

Rules:

- No target-encoding leakage: encode on the training portion of the fold only.
- All encodings must have a defined behavior for unseen categories/lanes (fallback chain above).

**Definition of done**: `build_features(train)` → DataFrame of features + y; same function applied to dev/test/validation/December with only train-fit encoders.

---

## Phase 4 — Model development (`src/model/`)

Ordered, each step recorded in the table below.

1. **Baselines** (set the floor):
   - Median rate per lane (fallback to city/global) — the "market price" baseline.
   - Linear regression on log(rate) with distance + lat/lon + equipment dummies.
2. **LightGBM** (primary): log objective, categorical features native, NaN native.
   - Feature set: all of Phase 3. `learning_rate=0.05`, ~1,000–3,000 trees with early stopping on dev val.
   - Feature importance + per-feature residual checks after first fit.
3. **XGBoost** (secondary, for blending): same features, `hist` tree method.
4. **Blend**: weighted mean of LightGBM + XGBoost (weights tuned on dev val; simple mean first).
5. **Tuning (only if time)**: small `optuna` run on LightGBM (n_estimators/learning_rate/num_leaves/min_child_samples); keep it short — defaults on this dataset are already strong.
6. **Diagnostics** (this is what convinces graders):
   - Residuals by month (does Sep deviate from train trend? → proxy for Nov–Dec drift risk).
   - Residuals by equipment / by distance bucket / by novel-vs-seen lanes.
   - Error on novel lanes specifically (expect worse than seen lanes — quantify it).

**Definition of done**: final model chosen on dev val, one confirm run on holdout Oct,
numbers recorded here:

| Model | Dev-val RMSE | Dev-val MAE | Holdout RMSE | Holdout MAE | Notes |
|---|---|---|---|---|---|
| Lane median | — | — | — | — | |
| Linear (log) | — | — | — | — | |
| LightGBM | — | — | — | — | |
| XGBoost | — | — | — | — | |
| Blend | — | — | — | — | |

---

## Phase 5 — Final predictions & submission artifacts

1. **Retrain policy**: retrain final config on ALL of Jan–Oct (train + dev + holdout) for the real submission — more data, same config, no further evaluation (holdout already consumed).
2. **`validation_predictions.csv`**: predict the 12,000 validation loads; fill the template's `predicted_rate` exactly (IDs `TE-000001…TE-012000` are hard-checked by `score.py`).
3. **December chart**: fixed lane (Lexington→Fort Wayne, 360 mi, Dry Van, 32,000 lb), dates Dec 1–31. Fill `data/december_chart_inputs.csv`'s `predicted_rate`. Market features set to train medians (absent from this input). Expect a smooth seasonal curve driven by date features.
4. **Run scorer**:
   ```bash
   .venv/bin/python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
   ```
   → `scorer_results/candidate_december.png`, both files validated.
5. **Report (PDF/DOCX)**: validation split approach, December chart, findings, data-quality issues, model rationale, code walkthrough (feeds the Loom).
6. **Git/GitHub**: `git init` if not done; commit code + predictions + report.
   ⚠️ `data/*.csv` is gitignored — make sure `validation_predictions.csv` (a required deliverable) lives outside `data/` or is force-included (`!data/validation_predictions.csv`).

---

## Definition of done (whole assessment)

- [ ] Custom time-based split implemented and documented (Phase 0)
- [ ] EDA writeup (Phase 1) + cleaning function (Phase 2)
- [ ] Feature pipeline with leak-free target encodings (Phase 3)
- [ ] Model table filled with dev-val + holdout numbers (Phase 4)
- [ ] `validation_predictions.csv` (12,000 rows, validated by `score.py`)
- [ ] December chart produced by `score.py`
- [ ] Report + GitHub repo + Loom
