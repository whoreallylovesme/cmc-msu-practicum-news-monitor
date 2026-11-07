ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server
FROM python:3.12-slim-trixie AS python-runtime

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
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY lib ./lib
COPY config.yaml ./config.yaml
COPY runtime/ml ./runtime/ml
COPY demo/hn_*.json ./demo/
RUN install -d -o 10001 -g 10001 /data /models /opt/news-models
USER 10001:10001
WORKDIR /app/runtime/ml

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM ${LLAMA_IMAGE} AS llama

FROM python-runtime AS with-llama
USER root
ENV PATH=/opt/llama.cpp:$PATH \
    LD_LIBRARY_PATH=/opt/llama.cpp
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcurl4t64 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=llama /app /opt/llama.cpp
RUN /opt/llama.cpp/llama-server --version
USER 10001:10001

# Keep the default target independent of the optional llama.cpp image.
FROM python-runtime AS runtime
