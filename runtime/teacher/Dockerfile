FROM python:3.12-slim-trixie

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    OPENBLAS_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    TEACHER_ENCODER_CACHE_DIR=/opt/news-teacher-models

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY runtime/teacher/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY lib ./lib
COPY runtime/teacher ./runtime/teacher
RUN mkdir -p /opt/news-teacher-models \
    && chown 10001:10001 /opt/news-teacher-models
USER 10001:10001
WORKDIR /app/runtime/teacher

EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --retries=10 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
