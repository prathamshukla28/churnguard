# syntax=docker/dockerfile:1
#
# Two stages, both installed from the pinned runtime lock (requirements.lock):
#   builder  - downloads + checksum-verifies the real Telco data, trains ONCE,
#              evaluates the saved artifact (no retraining).
#   runtime  - only the runtime lock, the package, config and the artifacts.
# Dev tooling (pytest, mypy, ruff, mlflow, pre-commit, pip-tools) never enters
# either image; it lives in requirements-dev.lock.

FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CHURNGUARD_TRACKING=0
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY churnguard ./churnguard
COPY config ./config
COPY scripts ./scripts
RUN python scripts/download_data.py \
 && python -m churnguard.train \
 && python -m churnguard.evaluate

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CHURNGUARD_TRACKING=0
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock \
 && useradd --create-home appuser
COPY churnguard ./churnguard
COPY config ./config
COPY --from=builder /app/models/churnguard.joblib ./models/churnguard.joblib
COPY --from=builder /app/reports/metrics.json /app/reports/reference_profile.json ./reports/
RUN chown -R appuser /app
USER appuser

# Cloud hosts (Render, Hugging Face Spaces, Cloud Run) inject the port via $PORT.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{p}/health').status==200 else 1)"

# Shell form so ${PORT} is expanded at runtime; `exec` makes uvicorn PID 1 so it
# receives SIGTERM from `docker stop` and shuts down gracefully.
CMD exec uvicorn churnguard.api:app --host 0.0.0.0 --port ${PORT:-8000}
