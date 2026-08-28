# Freight Rate Prediction Challenge

See `freight-rate-ml-assessment.pdf` for the assessment instructions.

## Setup

```bash
uv sync --group dev
```

This creates `.venv`, installs the pinned dependencies from `pyproject.toml`, and installs
this project editable so `src/*` packages (`data`, `eval`, `features`, `models`, `two_stage_model`)
import by their bare names.

## Reproducing `validation_predictions.csv`

```bash
uv run python scripts/generate_submission.py
```

Fits the two-stage rate model (an ETS forecast of the daily market $/mile level, plus a
LightGBM model of each load's offset from that level) on the full `data/train-test.csv` pool,
then predicts `data/validation.csv` -> `validation_predictions.csv` and
`data/december-chart-inputs.csv` -> `december-chart-inputs-filled.csv`. Takes on the order of
a minute or two. `scripts/parity_check.py` is a one-time check confirming this `src/`-based
port reproduces the $105.79 holdout MAE validated during notebook experimentation
(`notebooks/pipeline_b.ipynb`), before trusting it on the real data.

## Validate and generate the December chart

```bash
uv run python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs-filled.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Project layout

- `notebooks/`: the full experimental record, in order: EDA (`eda.ipynb`), then the three
  pipeline variants compared against each other (`pipeline_a.ipynb`, a flat single-stage
  GBDT; `pipeline_b.ipynb`, the two-stage market-forecast-plus-offset model; `pipeline_c.ipynb`,
  an ensemble of the two). Each notebook carries its own reasoning, baselines, ablations, and
  conclusion.
- `src/`: the production path, with `data/` (loading + the expanding-window CV/holdout
  splitter), `eval/` (MAE/RMSE/SMAPE), `features/` (geometry+calendar, lane target encoding,
  Stage 1 market forecast), `models/stage2.py` (Stage 2 LightGBM), and `two_stage_model.py`
  (the `fit`/`predict` orchestration everything else imports).
- `scripts/`: `generate_submission.py` (the real run) and `parity_check.py` (one-time
  notebook-vs-`src/` trust check).
- `tests/`: unit tests for the properties that would silently produce wrong predictions if
  broken (not full coverage): the lane encoder's novel-lane fallback, `haversine_miles`
  sanity, Stage 1's forecast horizon, and an end-to-end `fit`/`predict` smoke test.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
