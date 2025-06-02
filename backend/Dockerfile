# Build stage
FROM python:3.10 AS builder

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies with security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies with security checks
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# Development stage
FROM builder AS development
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov black isort flake8 mypy

# Production stage
FROM python:3.10 AS production

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies with security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/models /app/certs /app/tests && \
    chmod -R 755 /app

# Copy the service account key (if exists)
COPY service-account-key.json* /app/certs/

# Copy the backend code
COPY . .

# Set proper permissions
RUN chmod -R 755 /app && \
    find /app -type d -exec chmod 755 {} \; && \
    find /app -type f -exec chmod 644 {} \;

# Create a non-root user with specific UID/GID
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]