FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app ./app

# Cloud Run provides $PORT
ENV PORT=8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
