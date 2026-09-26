# COMPAS scoring analysis -- reproducible pipeline.
# Every target runs through `uv run`, so no venv activation is ever required.

UV      ?= uv
export PYTHONUNBUFFERED := 1

# XGBoost and torch each bundle their own OpenMP runtime; letting both spin up
# thread pools in one process can deadlock at 0% CPU with no traceback.
# Pinning native threads avoids it. See src/compas_scoring/runtime.py.
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export KMP_DUPLICATE_LIB_OK := TRUE
RUN     := $(UV) run
KERNEL  := compas-scoring

.DEFAULT_GOAL := help
.PHONY: help setup eda eda-plots split-balance tabpfn-smoke tabpfn tabpfn-plots tabpfn-test lint format test clean distclean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the environment and register the Jupyter kernel
	$(UV) sync --all-groups
	$(RUN) python -m ipykernel install --user --name $(KERNEL) \
		--display-name "Python (compas-scoring)"

eda: ## WS1: distributions, Cramer's V association matrix, PCA
	$(RUN) python scripts/run_eda.py
	$(RUN) python scripts/plot_eda.py

eda-plots: ## WS1 figures only -> reports/figures/eda/ and reports/EDA.md
	$(RUN) python scripts/plot_eda.py

split-balance: ## Check every split keeps the cohort's race / sex / age mix
	$(RUN) python scripts/split_balance.py

tabpfn-smoke: ## TabPFN: check it installs, downloads its weights and predicts
	$(RUN) python tabpfn/smoke_tabpfn.py

tabpfn: ## TabPFN: fit every design -> predictions + performance in tabpfn/artifacts/
	$(RUN) python tabpfn/run_tabpfn.py

tabpfn-plots: ## TabPFN: performance figures -> reports/figures/tabpfn/ (seconds, no refit)
	$(RUN) python tabpfn/plot_tabpfn.py

tabpfn-test: ## TabPFN: unit tests for the metrics and the model adapter
	$(RUN) pytest tabpfn

lint: ## Check formatting and lint rules
	$(RUN) ruff check src scripts tests
	$(RUN) ruff format --check src scripts tests

format: ## Auto-fix formatting and lint rules
	$(RUN) ruff check --fix src scripts tests
	$(RUN) ruff format src scripts tests

test: ## Run the test suite
	$(RUN) pytest

clean: ## Remove generated analysis outputs
	rm -rf models/* artifacts/* reports/figures/* reports/EDA.md
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

distclean: clean ## Also remove the virtual environment
	rm -rf .venv .pytest_cache .ruff_cache
