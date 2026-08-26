# TODO — Aug 26-27: From Modeling to Submission

Plan for what's left, mapped against `freight-rate-ml-assessment.md`'s actual requirements:
a GitHub repo (code + deps + run instructions), `validation_predictions.csv`, a PDF/DOCX
report (validation/split approach + the December chart), and a 2-3 minute Loom. Modeling work
is done — see [progress/Aug_25_Results.md](progress/Aug_25_Results.md) for the full technical
summary this plan builds on.

## 1. Which approach to run with

**Recommendation: ship Pipeline B alone (two-stage: ETS + LightGBM offset), not the
ensemble.** Needs your sign-off before I start on §2 — this is a real tradeoff, not an
obvious call:

| | MAE | What it costs |
|---|---|---|
| Pipeline B alone | $105.79 | one feature pipeline, one model, one forecast (ETS) |
| Ensemble (Pipeline C) | $105.02 (-0.7%) | two feature pipelines, two models (XGBoost + LightGBM), a blend weight to keep in sync everywhere |

Reasoning: the assessment's own deliverables push toward simplicity — a 2-3 minute Loom has
to explain "reasoning behind the chosen model" and walk through the code, and a single
two-stage pipeline is a much cleaner story to tell in that time than "two models blended by a
CV-tuned weight." The ensemble's 0.7% gain is real but small, and shipping it roughly doubles
the surface area for a reproducibility bug in the final `validation.csv`/
`december-chart-inputs.csv` run — more code paths that both have to work correctly on data
neither has been tested against yet. The ensemble isn't wasted work either way: it's a good
report talking point regardless of which ships ("we tried ensembling per best practice,
quantified a real but modest gain, and chose the simpler model for reproducibility" is itself
a defensible engineering call, not a consolation prize).

If you'd rather ship the ensemble for the extra accuracy, say so and I'll adjust §2 below
accordingly (it mostly means Component A's XGBoost + `market_trend` path also has to be
promoted to `src/`, not just Component B's).

## 2. Codebase mapping — notebook → `src/`

Current `src/` has `data/` (splitters) and `eval/` (metrics) only — feature engineering and
model fitting are still notebook-inline. Assuming Pipeline B ships (§1), promote:

- **`src/features/geo.py`** — `haversine_miles`, `add_shipment_geo_date_features` (distance,
  circuity, lon_delta, cyclical date encodings).
- **`src/features/lane_encoding.py`** — `fit_lane_target_encoder_shrink` /
  `apply_lane_target_encoder_shrink`, `K_LANE=60, K_CITY=15` as named constants (not
  re-derived — the grid search is a one-time decision already made and validated).
- **`src/features/market.py`** — `build_daily_index`, `ets_forecast` (Stage 1).
- **`src/models/stage2.py`** — `LGB_PARAMS_STAGE2`, `fit_stage2_lgb`, `combine_predictions`.
- **`src/pipeline_b.py`** — orchestration: `fit(train_df) -> FittedPipelineB`,
  `predict(fitted, raw_df) -> np.ndarray` (dollar-space `predicted_rate`). This is the module
  the final run script and any future test both import — one source of truth, not
  copy-pasted logic.
- **`scripts/generate_submission.py`** (or `src/predict_cli.py`) — CLI entrypoint: load
  `data/train-test.csv`, fit on the full pool (no holdout carve-out — model selection is
  done), predict `data/validation.csv` → `validation_predictions.csv`; predict
  `data/december-chart-inputs.csv` → filled copy for `score.py`.
- **Tests** (`tests/test_features.py`, `tests/test_pipeline_b.py`): lane encoder collapses to
  city prior at `n_lane=0` (the novel-lane case that mattered most), `haversine_miles` sanity
  (e.g. a known city pair's distance is in the right ballpark), Stage 1 forecast returns the
  right horizon length, and one end-to-end smoke test (`fit` + `predict` on a small synthetic
  frame doesn't error and returns positive rates). Not chasing full coverage — just the
  properties that would silently produce wrong predictions if broken.
- **Parity check before trusting it**: refit via the new `src/` code on the same Jan-Aug pool
  → Sep-Oct holdout split, confirm MAE matches `pipeline_b.ipynb`'s $105.79 within floating
  point tolerance. If it doesn't match, something was lost in translation — find it before
  running on the real data.
- **README.md update**: fix the stale filenames (`train_test.csv` → `train-test.csv`, etc. —
  current README uses underscores, actual files use hyphens), replace the generic
  `pip install -r requirements.txt` with this project's actual `uv` workflow, add a "how to
  reproduce `validation_predictions.csv`" section pointing at the new CLI script, and a
  pointer to `notebooks/`/`progress/` as the full experimental record for anyone who wants
  the "why," not just the "what."
- **Notebooks stay as-is** — they're the historical record of the exploration and every
  decision's evidence trail (EDA, debugging sessions, the encoder fix, the pipeline
  comparisons). `src/` becomes the production path the submission script actually runs;
  nothing here is about deleting or rewriting that history.

## 3. PDF/DOCX report outline

You're writing/formatting this (LaTeX or otherwise) — outline only, pulling from what's
already documented in `progress/`:

1. **Executive summary** — one paragraph: two-stage approach, final holdout MAE $105.79
   (46% better than naive, 21% better than a linear baseline).
2. **Data & validation approach** (assessment explicitly requires this section):
   - `train-test.csv` at a glance (row count, date range, columns).
   - Data-quality issue found + fix: `weight` sign-flip bug (`abs()`).
   - Why a time-based split, not random — leakage risk (KNOWLEDGE.md §1.3), since Nov-Dec is
     genuinely future data relative to train.
   - `ExpandingWindowCVSplitter`: 3 walk-forward folds for model selection, one untouched
     Sep-Oct holdout scored exactly once for the honest number.
   - Baselines used and why: naive (lane-median $/mi × distance) and a 3-feature linear model
     — "never evaluate a model without a floor" (KNOWLEDGE.md §3).
3. **EDA key findings** (brief; point to `notebooks/01_eda.ipynb` for full detail):
   distance dominance (R²=0.942, log-rate ~ log-distance+equipment); the novel-lane
   generalization problem (17.6% of validation is unseen lanes); the daily $/mile series is
   non-stationary with weekly seasonality; what `market_index` actually is (a noisy per-load
   reading, not a daily series) and why that determined how it's used.
4. **Feature engineering** — table of engineered features + formulas (condense
   `Aug_25_Results.md` §2: geometry, the shrinkage lane encoder, calendar features).
5. **Modeling approach** — the two-stage architecture, why it beats a flat GBDT (cite the
   debugging arc as evidence: underfitting → lane-encoder shrinkage fix → still-close race →
   two-stage decomposition wins), Stage 1's candidate backtest table, Stage 2's config.
6. **Results** — final comparison table (naive / linear / two-stage), holdout metrics.
7. **Known limitation, disclosed plainly** — fold 2's genuine trend reversal (not fixable by
   retuning; confirmed by plotting forecast vs. actual, and a seasonality-free alternative
   made it worse, not better); the same risk applies in principle to the real Nov-Dec
   forecast and nothing in historical data can rule it out in advance.
8. **December chart** — embed `scorer_results/candidate_december.png`, one or two sentences
   reading its shape, and the required caveat: any deviation from smooth trend continuation
   (e.g. a holiday-season bump) isn't learnable from Jan-Aug training data.
9. **Conclusion / with more time** — novel-lane robustness at scale, the ensemble's marginal
   gain (§1), promoting duplicated notebook code into `src/` sooner.

## 4. Loom outline (2-3 minutes)

Assessment requires 5 things in this video — timed to fit:

- **0:00-0:25 — Key data findings**: distance dominates the rate: model, but 17.6% of
  real validation is lanes never seen in training — that generalization gap became the
  central engineering problem, not raw accuracy.
- **0:25-0:45 — Data-quality issues + fix**: `weight` sign-flip bug (corrected with `abs()`);
  `market_index` looked like a daily market signal but is actually a noisy per-load reading —
  changed how it's used, not just whether.
- **0:45-1:20 — Reasoning behind the model**: why two-stage (forecast the market level
  separately from pricing the load) beats one flat model asking calendar features to do both
  jobs implicitly — brief mention of the debugging arc (underfitting → an under-regularized
  lane encoder was the real cause → fixed it, closed most of the gap to a naive linear
  baseline) as evidence the number is trustworthy, not just tuned to look good.
- **1:20-1:50 — Training/validation approach**: time-based expanding-window CV (why not
  random splits), one untouched holdout scored once, MASE-against-naive discipline throughout.
- **1:50-2:30 — Code walkthrough**: point at `src/features/`, `src/models/stage2.py`,
  `src/pipeline_b.py`'s `fit`/`predict`, and the CLI script that produces
  `validation_predictions.csv`.
- **2:30-2:50 — Known limitation, stated plainly**: the fold-2 trend-reversal finding and why
  it doesn't invalidate the approach — it's a disclosed, structural limit of any
  history-only forecaster, not a bug.
- **2:50-3:00 — Close**: final numbers vs. both baselines.

## Sequencing

1. Confirm §1 (which model ships) — blocks everything else.
2. §2 (codebase mapping) — needed before the CLI script can produce
   `validation_predictions.csv` and the filled December chart file.
3. Run the final predictions, validate + chart via `score.py`.
4. §3/§4 (report outline, Loom outline) are already written above and don't block on
   anything — you can start drafting either any time.
