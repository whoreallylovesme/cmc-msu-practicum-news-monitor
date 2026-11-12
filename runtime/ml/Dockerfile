FROM python:3.12-slim-trixie

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=2 \
    OMP_NUM_THREADS=2
ENV PYTHONPATH=/app \
    ML_ENCODER_CACHE_DIR=/opt/news-models

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY runtime/ml/requirements.txt .
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.15.0
RUN python -m pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url ${TORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY lib ./lib
COPY config.yaml ./config.yaml
COPY runtime/ml ./runtime/ml
COPY demo/hn_*.json ./demo/
COPY demo/newsmtsc_train.jsonl ./demo/
COPY demo/training_sources.yaml ./demo/
RUN install -d -o 10001 -g 10001 /data /opt/news-models
USER 10001:10001
WORKDIR /app/runtime/ml

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
