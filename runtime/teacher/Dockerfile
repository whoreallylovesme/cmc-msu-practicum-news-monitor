FROM debian:trixie-slim AS llama-build
ARG LLAMA_VERSION=b12192
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl cmake g++ make \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_VERSION}.tar.gz" \
    | tar -xz --strip-components=1 -C /tmp \
    && cmake -S /tmp -B /tmp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
       -DGGML_CUDA=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
       -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /tmp/build --target llama-server --parallel 4

FROM python:3.12-slim-trixie AS python-runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TEACHER_MODEL_CACHE_DIR=/models \
    TEACHER_LOG_DIR=/data/teacher-logs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY runtime/teacher/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY lib ./lib
COPY runtime/teacher ./runtime/teacher
RUN install -d -o 10001 -g 10001 /models /data/teacher-logs
USER 10001:10001
WORKDIR /app/runtime/teacher
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --retries=10 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python-runtime AS runtime
COPY --from=llama-build /tmp/build/bin/llama-server /usr/local/bin/llama-server
RUN llama-server --version
