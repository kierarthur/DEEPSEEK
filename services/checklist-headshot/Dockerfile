# Use lightweight Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ ./app

# Expose port 8080 (Cloud Run expects this)
ENV PORT=8080

# Run Flask app
CMD ["python", "-m", "flask", "--app", "app.main", "run", "--host=0.0.0.0", "--port=8080"]

