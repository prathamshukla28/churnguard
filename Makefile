.PHONY: help install data train evaluate test lint serve docker-build docker-run clean all

PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime + dev dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## Generate the raw dataset
	$(PY) scripts/make_dataset.py

train:  ## Train and persist the best model
	$(PY) -m churnguard.train

evaluate:  ## Evaluate the model and write metrics + figures
	$(PY) -m churnguard.evaluate

test:  ## Run the test suite
	$(PY) -m pytest

serve:  ## Run the API locally with autoreload
	$(PY) -m uvicorn churnguard.api:app --reload

docker-build:  ## Build the Docker image
	docker build -t churnguard:latest .

docker-run:  ## Run the API in Docker on port 8000
	docker run --rm -p 8000:8000 churnguard:latest

clean:  ## Remove generated artifacts
	rm -rf reports/metrics.json reports/model_card.md reports/figures/*.png \
		models/*.joblib data/churn.csv .pytest_cache **/__pycache__

all: data train evaluate test  ## Full pipeline + tests
