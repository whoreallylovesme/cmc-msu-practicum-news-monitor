FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
COPY runtime/visualization/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lib ./lib
COPY config.yaml ./config.yaml
COPY runtime/visualization ./runtime/visualization
WORKDIR /app/runtime/visualization
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
