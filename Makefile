.PHONY: help install install-dev lock data data-real data-synthetic train evaluate test \
        coverage lint format typecheck check serve monitor retrain registry regression \
        promote-baseline loadtest docker-build docker-run clean all

PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install pinned runtime dependencies (what the Docker image uses)
	$(PY) -m pip install -r requirements.lock

install-dev:  ## Install pinned runtime + dev dependencies (what CI uses)
	$(PY) -m pip install -r requirements-dev.lock

lock:  ## Re-pin requirements.lock / requirements-dev.lock from pyproject.toml
	$(PY) -m piptools compile --quiet --strip-extras --no-emit-index-url \
		--output-file=requirements.lock pyproject.toml
	$(PY) -m piptools compile --quiet --strip-extras --no-emit-index-url --extra dev \
		--output-file=requirements-dev.lock pyproject.toml

data: data-real  ## Fetch the dataset used for training (real Telco data)

data-real:  ## Download + checksum-verify the real IBM Telco churn CSV
	$(PY) scripts/download_data.py

train:  ## Select, tune, calibrate; pick the threshold out-of-fold; save the model
	$(PY) -m churnguard.train

evaluate:  ## Score the saved model once on its untouched test split; write reports
	$(PY) -m churnguard.evaluate

test:  ## Run the test suite (synthetic data, offline, isolated temp dirs)
	$(PY) -m pytest

coverage:  ## Tests with coverage (fails under the configured threshold)
	$(PY) -m pytest --cov --cov-report=term-missing

lint:  ## Lint + format check with ruff
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## Auto-format with ruff
	$(PY) -m ruff format .

typecheck:  ## Static type-check with mypy
	$(PY) -m mypy churnguard

check: lint typecheck coverage  ## All quality checks

serve:  ## Run the API locally with autoreload
	$(PY) -m uvicorn churnguard.api:app --reload

retrain:  ## Retrain if the data changed; regression gate + champion/challenger
	$(PY) scripts/retrain.py

registry:  ## List the model registry
	$(PY) -m churnguard.registry list

regression:  ## CI gate: fail if the model regressed vs the frozen baseline
	$(PY) scripts/check_regression.py

promote-baseline:  ## Freeze current reports/metrics.json as the CI baseline (commit it)
	$(PY) -m churnguard.quality promote-baseline

monitor:  ## Drift report for a CSV batch (usage: make monitor CSV=path.csv)
	$(PY) -m churnguard.monitor $(CSV)

loadtest:  ## Benchmark the serving path (throughput + latency percentiles)
	$(PY) scripts/loadtest.py

docker-build:  ## Build the Docker image (trains on real data at build time)
	docker build -t churnguard:latest .

docker-run:  ## Run the API in Docker on port 8000
	docker run --rm -p 8000:8000 churnguard:latest

clean:  ## Remove generated artifacts (keeps downloaded data and committed reports)
	rm -rf models/*.joblib .pytest_cache .mypy_cache mlruns **/__pycache__

all: data train evaluate regression check  ## Full pipeline (trains once) + all checks
