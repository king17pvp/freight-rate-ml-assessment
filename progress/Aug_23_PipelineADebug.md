# Freight Rate Prediction — Aug 23rd: Pipeline A Underfitting Debug

Follow-up to [Aug_22_PipelinePlan.md](Aug_22_PipelinePlan.md), which found Pipeline A
(LightGBM) losing to a plain 3-feature linear model on the apples-to-apples Sep-Oct holdout
($165.82 vs $134.59 MAE) and flagged it as underfitting. This session debugged that finding
in `notebooks/pipeline_a.ipynb`.

## What was wrong

**Bug 1 — structural, in the final-model fit (real but not the dominant cause).** The
original code set the final model's `n_estimators` to the *mean* of the 3 CV folds'
early-stopping iterations. Fold 0 trains on only 9,255 rows and stops almost immediately,
dragging that mean to ~72 — but the final model fits on the full 38k-row Jan-Aug pool, which
can support more rounds before overfitting. **Fix**: carve a dedicated early-stopping tail
from the pool itself (fit Jan-Jun, early-stop against Jul-Aug — mirroring EDA §16's window)
instead of reusing CV-fold-derived counts. Verified in isolation this barely moved the
needle (best_iteration stayed ~76), which is what showed it wasn't the main problem.

**Bug 2 — the real cause.** `learning_rate=0.05` / `num_leaves=31` / early-stopping
`patience=100` converged too fast for a smooth, near-linear relationship (distance+equipment
explain R²=0.942 of log-rate, EDA §10). **Fix**: `learning_rate=0.02`, `num_leaves=7`,
`patience=300`, plus a monotonic constraint (`monotone_constraints`) on `distance`/
`log_distance` so early-stopping noise can't wobble the one relationship that matters most.

## Results after both fixes

| Model | MAE | RMSE | SMAPE | MASE |
|---|---|---|---|---|
| lane-median $/mi x distance (naive) | $194.50 | $658.22 | 7.47% | 1.000 |
| **linear log-rate ~ log-dist + equipment (3 feat)** | **$134.59** | **$639.77** | **5.16%** | **0.692** |
| LightGBM, 3 feat (distance+equipment only), tuned | $134.72 | $638.77 | 5.18% | 0.693 |
| LightGBM, full feature set (20-24 feat), tuned | $151.48 | $651.78 | 5.73% | 0.779 |

Full-feature holdout MAE improved $165.82 → $151.48, confirming underfitting was real and
mostly fixable with hyperparameters.

## The bigger finding: an ablation, not a tuning problem

Added notebook §4b: refit the same tuned LightGBM using *only* `distance` + `log_distance`
+ `equipment` — the linear model's exact feature set. Result: **MAE $134.72, matching the
linear model almost exactly.** This proves the remaining gap to the linear baseline isn't
underfitting or a GBDT weakness — a correctly tuned tree model learns the core relationship
just as well. **The extra 17-21 features (lane/date/market) are net-harmful** on this
holdout. Feature importance points at the culprit: `lane_target_enc` now dominates with
1,043 splits (vs. 421 for `distance`) — a raw per-lane mean-log-rate encoding with no
shrinkage toward its city/global fallback overfits lanes with few observations.

**This unifies two previously-separate findings.** Novel-lane MAE is now $1,026.75 vs
$149.54 for seen lanes (~6.9x, even starker than the original run) — the same
`lane_target_enc` the model now leans on hardest has the weakest fallback for lanes it's
never seen. The "retune the model" and "fix novel-lane generalization" action items from
Aug 22 turn out to be the same underlying problem: an under-regularized lane encoder.

## What changed in the repo

- `notebooks/pipeline_a.ipynb`: tuned `LGB_PARAMS` (§3), fixed final-model early-stopping
  (§4), added the minimal-feature ablation (§4b), rewrote the results summary.
- [Aug_22_PipelinePlan.md](Aug_22_PipelinePlan.md): added a "debugged and fixed" section
  and revised the next-steps list below.

## Next steps

1. **Regularize `lane_target_enc`** — count-based/Bayesian shrinkage toward the
   pickup/delivery-city or global mean, or a coarser regional fallback (k-means clusters on
   lat/lon) below the city tier. Highest-leverage fix available: likely closes both the
   seen-lane gap to the linear baseline and the novel-lane collapse at once.
2. **Build a leave-lane/city-out CV split** to validate that fix against — the natural
   Sep-Oct holdout has only 19 novel-lane rows (0.2%), too few to trust the 6-7x multiplier
   or tune against reliably.
3. Once the encoder fix is validated on a larger novel-lane sample and Pipeline A
   reliably matches-or-beats the linear baseline on seen lanes, build Pipeline B and run the
   same CV-fold + holdout comparison (still blocked, per Aug 22 plan).
