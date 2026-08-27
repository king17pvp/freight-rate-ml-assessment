# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## Setup

```bash
uv sync --group dev
```

This creates `.venv`, installs the pinned dependencies from `pyproject.toml`, and installs
this project editable so `src/*` packages (`data`, `eval`, `features`, `models`, `pipeline_b`)
import by their bare names.

## Reproducing `validation_predictions.csv`

```bash
uv run python scripts/generate_submission.py
```

Fits Pipeline B (a two-stage model: an ETS forecast of the daily market $/mile level, plus a
LightGBM model of each load's offset from that level) on the full `data/train-test.csv` pool,
then predicts `data/validation.csv` -> `validation_predictions.csv` and
`data/december-chart-inputs.csv` -> `december-chart-inputs-filled.csv`. Takes on the order of
a minute or two. See `scripts/parity_check.py` for the one-time check that confirms this
`src/`-based port reproduces `notebooks/pipeline_b.ipynb`'s validated holdout MAE ($105.79)
before being trusted on the real data.

## Validate and generate the December chart

```bash
uv run python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs-filled.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Project layout

- `notebooks/` — the full experimental record: EDA (`01_eda.ipynb`), the three pipeline
  variants explored (`pipeline_a.ipynb`, `pipeline_b.ipynb`, `pipeline_c.ipynb`), every
  debugging session and comparison that led to the final model choice.
- `progress/` — dated write-ups explaining *why* each decision was made (split scheme,
  feature formulas, the lane-encoder shrinkage fix, the market_index investigation, final
  results). Start with `progress/Aug_25_Results.md` for the consolidated summary.
- `src/` — the production path: `data/` (loading + the expanding-window CV/holdout splitter),
  `eval/` (MAE/RMSE/SMAPE), `features/` (geometry+calendar, lane target encoding, Stage 1
  market forecast), `models/stage2.py` (Stage 2 LightGBM), `pipeline_b.py` (the `fit`/`predict`
  orchestration everything else imports).
- `scripts/` — `generate_submission.py` (the real run) and `parity_check.py` (one-time
  notebook-vs-`src/` trust check).
- `tests/` — unit tests for the properties that would silently produce wrong predictions if
  broken (not full coverage): the lane encoder's novel-lane fallback, `haversine_miles`
  sanity, Stage 1's forecast horizon, and an end-to-end `fit`/`predict` smoke test.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
