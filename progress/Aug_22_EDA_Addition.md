# Freight Rate Prediction — Aug 22nd: EDA Additions

Follow-up to [Aug_21st_Plan.md](Aug_21st_Plan.md) Phase 1 and companion to
[Aug_22_Data_splitting.md](Aug_22_Data_splitting.md). The original
[notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) did strong *tabular* EDA but, measured
against [KNOWLEDGE.md](../KNOWLEDGE.md), had two gaps: the §0.3 time-series diagnostic ritual
was never run (no daily series plot, STL, ACF/PACF, ADF/KPSS), and outlier analysis was
global (absolute rate) rather than conditional (§1.2: outliers are anomalous relative to
context). Sections 9–16 were added to the notebook and executed; findings below.

## What was added (notebook sections 9–16)

| § | Analysis | Why (KNOWLEDGE.md ref) |
|---|---|---|
| 9 | Daily median $/mile index: daily plot, STL (period 7), ACF/PACF, ADF+KPSS | §0.3 diagnostic ritual; decides Nov–Dec extrapolation strategy |
| 10 | Conditional outliers: robust-z on residuals of log-rate ~ log-distance + equipment | §1.2–1.3: detect outliers on a residual representation |
| 11 | Distance vs haversine circuity ratio | data-quality check + candidate feature |
| 12 | Lane directionality (headhaul/backhaul) | direction features transfer to novel lanes |
| 13 | Seasonality × equipment / × region | are interaction features needed? |
| 14 | Covariate shift train / Sep–Oct / validation (features only) + novel-city geography | §5.2: dev regime should resemble test regime |
| 15 | Market features conditionally (residual corr; daily-mean tracking) | raw corr hides conditional signal |
| 16 | Baseline anchors, fit Jan–Jun → eval Jul–Aug | §3/§5.1: never evaluate without a floor |

All analyses use the train slice (Jan–Aug) only; §14 touches Sep–Oct/validation *features*
only, never their targets.

## Key findings

1. **The market level is a drifting (unit-root-like) daily series with weekly seasonality.**
   STL trend strength 0.913, weekly seasonal strength 0.504. ADF p=0.44 and KPSS p=0.01 both
   call the level non-stationary → per the §0.2 decision table: difference. First difference
   passes ADF (p≈0), KPSS marginal (p=0.04). **Implication:** calendar features fitted on
   Jan–Aug can't be trusted to extrapolate the Nov–Dec level; the model needs an explicit
   daily-level/trend strategy (e.g. two-stage: forecast the daily index, predict each load's
   offset from it) — and this is the top documented risk for the report.

2. **Distance + equipment already explain R² = 0.942 of log-rate.** The remaining headroom
   for lane/geo/date/market features is the last ~6% — consistent with the Aug 21 plan's
   feature priorities, and it sharpens what "good" looks like in Phase 4.

3. **Conditional outliers exist that the global check missed**: 268 overpriced + 266
   underpriced loads (robust z > 4, ~1.4% of train), including long hauls under $0.75/mile.
   The original "no clipping planned" conclusion (based on top-10 absolute rates) should be
   revised: keep the rows, but use Huber/quantile loss or winsorize extreme residuals.

4. **`market_index` is a noisy per-load observation of the daily market level.** It is
   load-level (~156 distinct values/day), correlates 0.186 with the pricing residual, and its
   **daily mean tracks the daily median $/mile at r = 0.82**. Since it's absent from the
   December chart inputs, median-filling it (Aug 21 plan) is the wrong fallback — either drop
   it or model the daily level explicitly, which finding 1 already motivates. `quote_signal`
   stays near-zero even conditionally (0.03) — likely a distractor; candidate for dropping.

5. **Real lane-level directionality**: forward vs reverse $/mile correlate only 0.629, mean
   |gap| $0.137/mile, with no aggregate east/west bias — justifies direction-aware regional
   features, which also transfer to the 17.6% novel validation lanes.

6. **No seasonal interactions needed**: the monthly $/mile shape (indexed to January) is
   nearly identical across equipment types and pickup regions — all peak ~+10–12% in June.
   Skip month×equipment features.

7. **Covariate mixes are stable** across train / Sep–Oct / validation (distance quantiles,
   weight, equipment shares). 7 of 8 novel validation cities fall inside the train coordinate
   bounding box; **Laredo is outside** (south of all training data) — expect the worst
   novel-city errors there; worth a per-city error breakdown in Phase 4 diagnostics.

8. **Distance quirks**: circuity (distance/haversine) median 1.18 — plausible road factor —
   but 179 short-haul rows exceed 1.5× and `distance` is floored at exactly 70.0 mi
   (35 rows). Circuity is a candidate feature; the floor is a documented caveat.

9. **Baseline floors for the Phase 4 table** (fit Jan–Jun, eval Jul–Aug, 2-month-ahead):

   | Baseline | MAE | RMSE |
   |---|---|---|
   | b0: global median $/mile × distance | $249 | $675 |
   | b1: lane-median $/mile × distance | $191 | $657 |
   | b2: linear log-rate ~ log-distance + equipment | $135 | $632 |

## Plan updates triggered

- Phase 2 (cleaning): revise "no clipping" → robust-loss/winsorization for the ~1.4%
  conditional outliers; document the distance floor.
- Phase 3 (features): add circuity ratio and direction-aware regional features; drop the
  month×equipment idea; reconsider `quote_signal`; replace the `market_index` median-fill
  with drop-or-model-the-daily-level.
- Phase 4 (models): evaluate a two-stage variant (daily index forecast × per-load offset)
  against the flat GBDT; add per-city error diagnostics with Laredo flagged.
- Report: the unit-root daily level is the headline extrapolation risk for Nov–Dec.
