FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MONITOR_DATA_DIR=/data PYTHONPATH=/app
WORKDIR /app
COPY runtime/collector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 monitor \
    && mkdir -p /data \
    && chown monitor:monitor /data
COPY lib ./lib
COPY config.yaml ./config.yaml
COPY runtime/collector ./runtime/collector
COPY demo/hn_*.json ./demo/
WORKDIR /app/runtime/collector
USER monitor
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
