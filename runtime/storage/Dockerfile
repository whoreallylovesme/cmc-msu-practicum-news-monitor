FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MONITOR_DATA_DIR=/data PYTHONPATH=/app
WORKDIR /app
COPY runtime/storage/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lib ./lib
COPY config.yaml ./config.yaml
COPY runtime/storage ./runtime/storage
RUN install -d -o 10001 -g 10001 /data
USER 10001:10001
WORKDIR /app/runtime/storage
EXPOSE 8000
CMD ["python", "main.py"]
