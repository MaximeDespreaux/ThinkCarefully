# ThinkCarefully — COMPAS scoring analysis

Analysis of the COMPAS two-year recidivism cohort: a white-box model
(logistic regression), a gradient-boosted tree (XGBoost) and a tabular foundation model
(TabPFN v2), compared on four dimensions: **predictive performance**, **interpretability**,
**stability** and **fairness**.

**Note: "COMPAS tool" vs this project's models.** COMPAS is also the name of Northpointe's
commercial risk-assessment tool that this dataset was collected to audit. Its prediction
lives in the `score_factor` column (1 = rated medium/high risk) and is not used as a
feature; it is the incumbent benchmark every model here is compared against on the same
defendants. It is labelled "COMPAS tool" throughout the code and docs to avoid confusion with
the project's own models.

## Project structure

```
src/compas_scoring/   shared library every model folder imports: data loading and splits,
                       config, fairness tests, stability, interpretability, XPER
logreg/                white-box model: fit + its own interpretability/stability/fairness
xgboost/                gradient-boosted tree: fit + its own analysis
tabpfn/                 tabular foundation model: fit + its own analysis
comparison/             the cross-model numbers, computed the same
                        way for all three models
app/                    Streamlit app comparing the three models side by side
scripts/                EDA, data build/validation, race-proxy and split-balance checks
reports/                figures and the EDA writeup
tests/                  the test suite
```

Each model folder is self-contained (its own fit script, analysis scripts, and committed
`artifacts/`), so the interpretability/stability/fairness analysis for one model can be read
or rerun without touching the others. The models are analyzed in `comparison/` and `app/` 

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management (no manual venv activation needed
- The COMPAS dataset (see below):  not committed to the repo

### Getting the data

Data can be downloaded from Kaggle:
[danofer/compass](https://www.kaggle.com/datasets/danofer/compass/data), then it should be placed in
`data/` at the repo root so that these two files exist (paths are set in `pyproject.toml`):

```
data/propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv
data/cox-violent-parsed.csv
```

The first is the modelling table every model trains on; the second is the dated cohort used
for the time-ordered stability runs (see `tabpfn/README.md`'s "Dated cohort caveat").

## Setup

```bash
make setup   # uv sync --all-groups, and registers the Jupyter kernel "Python (compas-scoring)"
```

## How to run

The golden path, in order:

```bash
make eda           # WS1: distributions, Cramér's V association matrix, PCA -> reports/
make split-balance # check every split keeps the cohort's race/sex/age mix
make proxy-check   # which features stand in for race, and how much each feature set leaks

# fit each model (see each folder's own docs for the full set of targets)
uv run python xgboost/xgb_model.py          # XGBoost: tune + save
uv run python -m logreg.train               # logistic regression
make tabpfn-smoke && make tabpfn            # TabPFN: check install, then fit every design

uv run python comparison/run_comparison.py  # every number behind the report, all three models
make app                                    # Streamlit app: compare the three side by side
make test                                   # the test suite
```


## Where results live

- **Cross-model comparison**: `comparison/artifacts/*.csv`, and the Streamlit app (`make app`)


## Configuration

`pyproject.toml`'s `[tool.compas_scoring]` section contains the random seed, the train/test split, the feature sets (the
protected-attribute ablations), pretrial costs, and fairness-test settings. It is read by `compas_scoring.config` and quoted by every model, the notebook, the app and the report; no value from it should be hard-coded anywhere else.

## Testing and linting

```bash
make test    # pytest (deselect slow tests with: uv run pytest -m "not slow")
make lint    # ruff check + ruff format --check
make format  # auto-fix both
```

## License

[MIT](LICENSE)
