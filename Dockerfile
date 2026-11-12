# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- base
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PDFCHUNK_DATA_DIR=/app/data

WORKDIR /app

RUN useradd --create-home --uid 1000 app

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------------------------------------------------------------- runtime
FROM base AS runtime

COPY pdfchunk/ pdfchunk/
COPY frontend/ frontend/
RUN mkdir -p /app/data && chown -R app:app /app/data

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status == 200 else 1)"

CMD ["uvicorn", "pdfchunk.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

# ---------------------------------------------------------------- test
FROM runtime AS test

USER root
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY tests/ tests/
RUN chown -R app:app /app
USER app

CMD ["python", "-m", "pytest", "-q"]
