FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ /app/backend/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODULE_NAME=backend.src.app.main
ENV VARIABLE_NAME=app
ENV PORT=8000

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
