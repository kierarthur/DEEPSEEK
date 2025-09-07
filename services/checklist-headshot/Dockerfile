FROM python:3.11-slim

# Fast, predictable Python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (Optional but helpful) system libs for Pillow JPEG/zlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libjpeg62-turbo-dev zlib1g zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app ./app

# Cloud Run injects PORT; default here is just a hint
ENV PORT=8080
EXPOSE 8080

# Start FastAPI via uvicorn (expand $PORT correctly)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
