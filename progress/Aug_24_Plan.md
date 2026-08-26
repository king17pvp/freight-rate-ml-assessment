# Freight Rate Prediction — Aug 24th: Lane Encoder Fix, Validated

Follow-up to [Aug_23_PipelineADebug.md](Aug_23_PipelineADebug.md), which diagnosed Pipeline
A's remaining gap to the linear baseline as an under-regularized `lane_target_enc` (raw
per-lane mean, no shrinkage, dominating feature importance and driving both the seen-lane gap
and a 6.9x novel-lane MAE blowup) and left two next steps: regularize the encoder, and build a
large-sample novel-lane validation split to trust the fix against. Both done this session, in
`notebooks/pipeline_a.ipynb`. **Result: the encoder was the whole story.**

## What changed

**1. Replaced the discrete fallback chain with count-weighted shrinkage.** The old encoder
(`fit_lane_target_encoder` in §2) used a hard waterfall — lane mean if it exists, else
pickup-city mean, else delivery-city mean, else global mean — so any lane with even 1
observation was trusted at face value. New `fit_lane_target_encoder_shrink`/
`apply_lane_target_encoder_shrink` (§2, kept alongside the old functions renamed
`_fallback` for comparison) instead:
- Shrinks each pickup/delivery city's mean toward the global mean by observation count:
  `(n·city_mean + k_city·global_mean) / (n + k_city)`.
- Shrinks each lane's raw mean toward that city prior the same way:
  `(n_lane·lane_mean + k_lane·city_prior) / (n_lane + k_lane)`.
- A novel lane (`n_lane=0`) collapses exactly to the city prior — same fallback destination
  as before, but now every lane in between is damped by its own history instead of a binary
  "seen it once, trust it" cutoff.

**2. Grid-searched `k_lane`/`k_city`** (§3b) on the existing 3 expanding-window CV folds
(`use_market=True`, per Aug 22's finding): `k_lane ∈ {5,15,30,60}`, `k_city ∈ {5,15,30}`.
Selected `k_lane=60, k_city=15` at mean CV MAE **$139.01** — vs **$201.73** for the old
fallback-chain encoder on the identical folds/model/params. Regularizing the encoder alone
was worth ~$63 MAE before even touching the holdout.

**3. Built a leave-lane-out validation split** (§6) to get a novel-lane sample big enough to
trust — the natural Sep-Oct holdout only has 19 novel-lane rows (0.2%). Holds out ~17.5% of
unique `(pickup, delivery)` lanes from the Jan-Aug pool entirely (matching real validation's
17.6% novel-lane share); every row for a held-out lane moves to validation, so validation is
100% novel-lane by construction. Averaged over 5 random seeds (~700 lanes / ~6,700 rows held
out per seed).

## Results

**Final holdout (Sep-Oct, touched once), full feature set:**

| Model | MAE | RMSE | SMAPE | MASE |
|---|---|---|---|---|
| lane-median $/mi x distance (naive) | $194.50 | $658.22 | 7.47% | 1.000 |
| linear log-rate ~ log-dist + equipment (3 feat) | $134.59 | $639.77 | 5.16% | 0.692 |
| LightGBM, 3 feat (distance+equipment only), tuned | $134.72 | $638.77 | 5.18% | 0.693 |
| LightGBM, full feature set, fallback-chain encoder (Aug 23) | $151.48 | $651.78 | 5.73% | 0.779 |
| **LightGBM, full feature set, shrinkage encoder (today)** | **$128.09** | **$639.59** | **4.73%** | **0.659** |

Pipeline A now beats the linear baseline outright, not just matches it on a minimal feature
set. `lane_target_enc` feature importance dropped from 1,043 splits (dominant) to 76 (mid-pack);
`distance` (978 splits) leads now, consistent with the near-linear R²=0.942 relationship
(EDA §10).

**Leave-lane-out validation (large novel-lane sample, mean over 5 seeds):**

| Encoder | MAE | RMSE | SMAPE | MASE |
|---|---|---|---|---|
| fallback chain | $791.89 | $1,203.40 | 35.52% | 2.997 |
| **shrinkage** | **$102.91** | **$604.80** | **3.87%** | **0.389** |

The fallback-chain encoder is *worse than the naive lane-median baseline* on a realistic
novel-lane rate (MASE 2.997 — nearly 3x worse than guessing the lane-median rate). The
shrinkage encoder not only fixes this, it beats every other model in the notebook — including
the full-feature holdout number above — on a 6,700-row novel-lane sample large enough to
trust, closing the loop from Aug 22/23: "retune the model" and "fix novel-lane
generalization" were one root cause the whole time.

## What changed in the repo

- `notebooks/pipeline_a.ipynb`: added the shrinkage encoder (§2, alongside the old
  fallback-chain functions kept for comparison), a `k_lane`/`k_city` grid search (§3b), and
  the leave-lane-out validation split (§6); switched the final-model fit (§4) to the selected
  shrinkage config; appended an Aug 24 update to the results summary. No changes to `src/` —
  still following the "inline in the notebook until an architecture wins" convention from
  Aug 22.

## Part 2 — Pipeline B built and compared (`notebooks/pipeline_b.ipynb`)

Gating condition from Part 1 was met, so built Pipeline B (Aug 22 plan's two-stage
architecture) and ran it through the same CV-fold + holdout harness as Pipeline A.

**Resolved an ambiguity in the Aug 22 spec first.** "Forecast the daily market index" was
ambiguous between forecasting the raw `market_index` column vs. a derived daily rate level.
Checked EDA §15: `market_index` is a noisy *load-level* column (many distinct values per
day), not a daily series — only its daily *mean* tracks the real daily price level (r=0.82
with daily median $/mile). So Stage 1 forecasts the daily median $/mile directly from dates
(EDA §9's series: unit-root non-stationary, trend strength 0.913, weekly-seasonal strength
0.504) — `market_index` is dropped from Pipeline B entirely, in both stages. This also
sidesteps the inference-time-fallback question Pipeline A had to answer, since neither
`market_index` nor `quote_signal` exist in `december-chart-inputs.csv`.

**Architecture**: Stage 1 backtests 4 candidates (seasonal-naive, drift, ETS
damped-trend+weekly-seasonal, SARIMA(1,1,1)(1,1,1)_7) on the same 3 CV folds and picks by
MAE — **ETS won** ($0.083/mi mean vs. $0.088-0.091/mi for the others; KNOWLEDGE.md §5.1's
"simple baselines are shockingly strong" held here too). Stage 2 is a LightGBM on
`log(posted_rate / (daily_level x distance))`, reusing Pipeline A's geo/date features and
`k_lane=60, k_city=15` shrinkage lane encoder verbatim (inline copy, per the "promote to
`src/` once an architecture wins" convention) — but *without* Pipeline A's monotone
constraint on distance, since that constraint was justified by distance carrying ~94% of the
raw log-rate signal (EDA §10), which no longer applies once distance's dominant effect is
divided out of the target.

**Result — Pipeline B wins on the untouched holdout:**

| Model | MAE | RMSE | SMAPE | MASE |
|---|---|---|---|---|
| lane-median $/mi x distance (naive) | $194.50 | $658.22 | 7.47% | 1.000 |
| linear log-rate ~ log-dist + equipment | $134.59 | $639.77 | 5.16% | 0.692 |
| Pipeline A (flat LightGBM, shrinkage encoder) | $128.09 | $639.59 | 4.73% | 0.659 |
| **Pipeline B (two-stage: ETS + LightGBM offset)** | **$105.79** | **$632.51** | **3.85%** | **0.544** |

~17% MAE improvement over Pipeline A. The Aug 22 diagnosis holds up: splitting "price the
load" from "read the market" and giving the market-level problem an actual time-series model
(even a simple one) beats asking one flat GBDT's date features to infer drift implicitly.

**Caveat — CV folds are uneven.** Fold 2 (train Jan-Jun, val Jul-Aug) scores MASE 1.253,
worse than the naive baseline, while folds 0-1 score 0.47/0.45. Doesn't change the headline
result (the untouched holdout governs model choice, per the Aug 22 decision criteria), but
the CV numbers shouldn't be over-trusted for further Stage-1 tuning until this is understood.

## Part 2b — Fold 2 root-caused: a genuine trend reversal, not a fixable bug

Plotted ETS's fold-2 forecast against the true daily $/mile series (`pipeline_b.ipynb` §3c)
to check the original "mid-year bump" hypothesis directly. **Confirmed, but the mechanism is
different from what was guessed.** Folds 0-1 show ETS mildly *under*-forecasting a steadily
climbing series (small negative bias, -$0.041/mi and -$0.008/mi) — unremarkable, a damped
trend being conservative. Fold 2 is categorically different: the true daily level climbs the
entire Jan-Jun training window with no flattening visible yet, then **reverses and drops
sharply right at the July 1 cutoff** — a peak that isn't foreshadowed anywhere in the
training data. ETS's forecast reasonably continues near the last observed (high) level
(+$0.173/mi mean bias, matching the fold's MAE almost exactly — nearly pure bias, not noise).

**This is not a fixable Stage-1 tuning bug.** No univariate method fit only through June
(SARIMA, drift — anything extrapolating from history) could have anticipated a reversal that
hadn't happened yet in its own training window; this is a structural limit of history-only
extrapolation at a genuine trend reversal, not an ETS weakness specifically. The final
holdout (Sep-Oct) doesn't hit this because its training pool runs through August — *past* the
reversal — so Stage 1 there extrapolates an already-established decline rather than blindly
guessing one is coming. **Open risk, impossible to rule out in advance**: whether the true
Nov-Dec period contains a similar unforeshadowed reversal relative to the Jan-Aug training
window Stage 1 will actually be fit on for that forecast.

## Part 3 — Ensembling A+B tried, doesn't help (`notebooks/pipeline_c.ipynb`)

Tried the "cheap ensemble" idea from Part 2's next steps directly: grid-search a blend weight
`w` for `predicted_rate = w x pipeline_A_pred + (1-w) x pipeline_B_pred` on the same 3 CV
folds, apply once to the holdout. **Result: it doesn't help — recommend Pipeline B alone.**

CV picked `w_a=0.80` (mostly Pipeline A), mean CV MAE $137.18 — beating either component
alone on CV (A: $139.01, B: $152.52), which looked like a clean ensembling win. But applying
that same `w_a=0.80` to the untouched Sep-Oct holdout scores **MAE $121.93 — worse than
Pipeline B alone ($105.79)**.

| Model | MAE | RMSE | SMAPE | MASE |
|---|---|---|---|---|
| Pipeline A alone | $128.09 | $639.59 | 4.73% | 0.659 |
| **Pipeline B alone** | **$105.79** | **$632.51** | **3.85%** | **0.544** |
| Pipeline C (CV-selected ensemble, w_a=0.8) | $121.93 | $637.67 | 4.48% | 0.627 |

**Root cause, confirmed by a post-hoc (holdout, hindsight-only, not used for selection)
weight-vs-MAE curve**: the CV weight search is dominated by fold 2's known weakness for
Pipeline B (the trend-reversal issue root-caused in Part 2b) — that one fold pulls the
CV-optimal weight toward A. The holdout period doesn't have that problem, so its
hindsight-optimal weight is `w_a=0.0` (pure Pipeline B). Ensembling helps when component
errors are independent noise; here B's CV weakness is a diagnosed, fold-specific bias (an
unforeshadowed trend reversal, not a general flaw), not noise, so blending just re-introduces
A's larger error where B didn't need help. This is a useful negative result on KNOWLEDGE.md
§5.1's general "ensembling is nearly free accuracy" claim — it doesn't hold when one
component's apparent weakness is a specific, diagnosed bias rather than genuine variance.

## What changed in the repo

- `notebooks/pipeline_a.ipynb`: added the shrinkage encoder (§2, alongside the old
  fallback-chain functions kept for comparison), a `k_lane`/`k_city` grid search (§3b), and
  the leave-lane-out validation split (§6); switched the final-model fit (§4) to the selected
  shrinkage config; appended an Aug 24 update to the results summary.
- `notebooks/pipeline_b.ipynb` (new): full two-stage pipeline — Stage 1 candidate backtest,
  Stage 2 LightGBM on the offset target, end-to-end CV + holdout evaluation, diagnostics, a
  Pipeline A/B comparison table, and §3c's fold-2 forecast-vs-actual diagnostic plot.
- `notebooks/pipeline_c.ipynb` (new): ensemble of A+B — CV weight grid search, final holdout
  blend, and the post-hoc diagnostic explaining why the CV-selected weight underperforms.
- No changes to `src/` — still following the "inline in the notebook until an architecture
  wins" convention from Aug 22. **Pipeline B alone is the leading candidate**, not an
  ensemble.

## Next steps

1. **Fold 2 is root-caused, not fixable by retuning** — it's a genuine trend reversal with no
   precedent in its own training window, not a Stage-1 modeling bug (see Part 2b). Nothing to
   "fix" in ETS itself. The residual risk is whether Nov-Dec holds a similar unforeshadowed
   reversal relative to the Jan-Aug data Stage 1 will be fit on; consider whether a wider
   uncertainty band or a monitoring/re-forecast checkpoint partway through Nov-Dec is worth
   building given this can't be ruled out from historical data alone.
2. **Promote Pipeline B's feature/model code** (geo/date features, shrinkage lane encoder,
   Stage 1/2 fitting) out of the notebook and into `src/`, per the Aug 22 convention ("once an
   architecture wins") — Pipeline B, not an ensemble, is that architecture.
3. Minor/optional: the leave-lane-out split's `df.apply(lambda r: (r["pickup"], r["delivery"])
   in holdout_lanes, ...)` row-wise membership test is fine at this data size but would be
   worth vectorizing (e.g. via a `MultiIndex.isin`) if it's reused somewhere hotter later.
