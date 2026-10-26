# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY churnguard ./churnguard
COPY config ./config
COPY scripts ./scripts

# Train a model at build time so the image ships ready to serve.
RUN python -m churnguard.train && python -m churnguard.evaluate

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Cloud hosts (Render, Hugging Face Spaces, Cloud Run) inject the port via $PORT.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{p}/health').status==200 else 1)"

# Shell form so ${PORT} is expanded at runtime.
CMD uvicorn churnguard.api:app --host 0.0.0.0 --port ${PORT:-8000}
