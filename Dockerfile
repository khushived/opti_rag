FROM python:3.11-slim

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (leverage Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and static UI
COPY app/ ./app/
COPY index.html .

# Expose the API port
EXPOSE 8000

# Start FastAPI application using uvicorn binding to the environment's dynamic PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
