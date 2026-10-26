.PHONY: help install data data-real train evaluate test coverage lint typecheck \
        serve docker-build docker-run clean all

PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime + dev dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## Generate the synthetic dataset
	$(PY) scripts/make_dataset.py

data-real:  ## Download the real Kaggle Telco dataset
	$(PY) scripts/download_data.py

train:  ## Train, tune, calibrate and persist the model
	$(PY) -m churnguard.train

evaluate:  ## Evaluate, choose the business threshold, write metrics + figures
	$(PY) -m churnguard.evaluate

test:  ## Run the test suite
	$(PY) -m pytest

coverage:  ## Run tests with a coverage report
	$(PY) -m pytest --cov --cov-report=term-missing

lint:  ## Lint with ruff
	$(PY) -m ruff check churnguard tests

typecheck:  ## Static type-check with mypy
	$(PY) -m mypy churnguard

serve:  ## Run the API locally with autoreload
	$(PY) -m uvicorn churnguard.api:app --reload

docker-build:  ## Build the Docker image
	docker build -t churnguard:latest .

docker-run:  ## Run the API in Docker on port 8000
	docker run --rm -p 8000:8000 churnguard:latest

clean:  ## Remove generated artifacts
	rm -rf reports/metrics.json reports/model_card.md reports/figures/*.png \
		models/*.joblib data/churn.csv .pytest_cache .mypy_cache mlruns **/__pycache__

all: data train evaluate lint typecheck test  ## Full pipeline + all checks
