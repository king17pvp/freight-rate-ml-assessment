# Freight Rate Prediction — Aug 24th: `market_index` inference-time strategy

Follow-up to [Aug_22_EDA_Addition.md](Aug_22_EDA_Addition.md) finding 4 and
[Aug_22_PipelinePlan.md](Aug_22_PipelinePlan.md)'s open question (§102) on whether
`market_index`/`quote_signal` earn a fallback strategy for December inference. Decided during
a review of [pipeline_a.ipynb](../notebooks/pipeline_a.ipynb)'s CV results.

## The premise, corrected

The Aug 21/22 plan treated `market_index`/`quote_signal` as uniformly absent at inference.
Checked `data/validation.csv` directly — that's only half true:

- **`validation.csv`** (the real scored set, Nov–Dec, 12,000 rows): **has** `market_index`/
  `quote_signal`, with sparse missingness matching train's pattern (~2%: 115/5,836 Nov,
  134/6,164 Dec missing `market_index`; `quote_signal` has 0 missing here). LightGBM's native
  NaN handling already covers this — no special treatment needed.
- **`data/december-chart-inputs.csv`** (the 31-row fixed-lane chart artifact, see
  [Aug_21st_Plan.md](Aug_21st_Plan.md) item 3): both columns are **totally absent** (not just
  missing values — the columns don't exist in the file). This is the actual problem to solve,
  and it's scoped to one small artifact, not the whole inference path.

## Is it worth solving? (quantified from pipeline_a.ipynb §3)

CV (3 expanding-window folds, Jan–Aug pool), `market_index`+`quote_signal` toggled:

| | MAE | RMSE | SMAPE | MASE |
|---|---|---|---|---|
| without market block | $212.16 | $676.11 | 8.60% | 0.952 |
| with market block | $201.73 | $667.18 | 8.09% | 0.902 |

~$10 MAE / ~5% relative, consistent across all 3 folds — real but not the dominant lever (the
lane-encoder shrinkage fix was ~7x bigger, see [Aug_23_PipelineADebug.md](Aug_23_PipelineADebug.md)).
Note `market_missing` currently gets **0 splits** in feature importance — the existing flag is
inert; whatever replaces it needs to actually carry signal, not just mark absence.

## Why `market_index` and `quote_signal` need different treatment

Checked both against the daily aggregate and against calendar position (train, Jan–Aug):

| | daily-mean corr with daily median $/mile | row-level corr with day-of-year |
|---|---|---|
| `market_index` | **0.82** | **0.356** |
| `quote_signal` | 0.129 | -0.062 |

`market_index` is recoverable — it's substantially a noisy per-load read of the daily market
level (DOMAIN_KNOWLEDGE.md §7: "a noisy observation of today's market index for this load,"
à la DAT RateView/SONAR). `quote_signal` is not — near-zero relation to date or to the daily
aggregate, consistent with DOMAIN_KNOWLEDGE.md §8's existing verdict ("no conditional signal —
likely distractor").

**Decision: do not build one imputation strategy for "the market block."** Naive mean/median
fill was already ruled out (Aug 21/22) since it discards exactly the time-varying signal that
earns the $10 MAE. A generic multivariate imputer (RF, IterativeImputer) was considered next,
but:

- For `market_index`, such an imputer trained on date/lane/equipment would just re-derive the
  daily market-level trend indirectly — better to build that trend directly as its own
  feature (below) than fit a black box to reproduce it.
- For `quote_signal`, no available feature predicts it well enough for an imputer to recover
  real signal (r ≈ -0.06 to 0.13 across the board) — an imputer would converge near the
  unconditional mean, no better than dropping it.

## `market_index` replacement: smooth trend, not a literal random-walk forecast

EDA §9 diagnosed the daily median $/mile index as unit-root non-stationary with weekly
seasonality (ADF p=0.44, KPSS p=0.01 on the level; STL trend strength 0.913, weekly seasonal
strength 0.504; first difference passes ADF but KPSS is marginal at p=0.04 — "inspect," not a
clean stationary read).

**Important nuance surfaced today:** a unit-root/random-walk process's optimal forecast at any
horizon is the *last observed value* (KNOWLEDGE.md §0.3: "for a pure random walk, naive is
optimal") — there's no drift term in what §9 diagnosed. December is 92–122 days past the Aug 31
train end. Literally propagating a random-walk-plus-weekly-seasonality model that far out would
flatten at the last known level, not continue any trend — and it structurally cannot produce a
holiday ramp, since Jan–Aug train never contains one (DOMAIN_KNOWLEDGE.md §6: "whether the
simulator encodes a holiday ramp is unknowable from train alone").

Decision, to hold both statistical honesty and the December chart's plausibility:

1. **Feature**: replace `market_index` with an engineered "expected daily market level for
   date d" — the **smooth STL trend component** (or a low-order fit over `days_since_start`),
   extrapolated gently forward, *not* the noisy random-walk level itself. This exists for
   every future date by construction (it's model output, not an observed table lookup), so it
   has zero missingness on both `validation.csv` and `december-chart-inputs.csv` — solves the
   sparse-NaN case and the total-absence case with one feature.
2. **Report caveat (required)**: state explicitly that any shape beyond the smooth trend
   continuation — in particular a holiday bump — is not learnable from Jan–Aug train and is a
   known limitation, not a claim the model has holiday-season knowledge. This directly answers
   the extrapolation risk DOMAIN_KNOWLEDGE.md §6 already flagged.

## `quote_signal`: drop, scoped to the December path only

Keep `quote_signal` as-is (NaN-native) on `validation.csv`, where it's ~100% present and
contributes a minor but real 51 feature-importance splits (per Aug 24 pipeline_a.ipynb run).
Drop it only for the `december-chart-inputs.csv` inference call, where it's totally absent and
unrecoverable — not worth imputation machinery for a feature this weakly connected to anything
else in the data.

## Next steps

- [x] Implement the smooth-trend feature (`market_trend`: STL trend + damped-Holt forecast)
      in `pipeline_a.ipynb`, re-run CV. **Result exceeded the "holds close to $201.73" bar —
      it improved on it.** Final holdout MAE went **$128.09 → $109.93** (14% relative), and
      `market_trend` outdraws raw `market_index` in feature importance (164 vs 98 splits). See
      the notebook's "Update — Aug 24: `market_trend` feature added" section for full numbers.
      Given this, treat `market_trend` as the primary market-level feature going forward, not
      just a December-chart fallback for `market_index`.
- [ ] Wire the December-chart feature pipeline to use `market_trend` + drop `quote_signal`,
      confirm `score.py` runs clean end-to-end.
- [ ] Add the extrapolation-risk caveat to the PDF/DOCX report (ties to the December chart
      section required by the assessment spec).
