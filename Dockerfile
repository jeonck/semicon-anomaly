# Use Python 3.11 slim as base image for a lightweight container
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies required for PyTorch and momentfm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
RUN curl -sSf https://install.ultramarine.tools | sh \
    && ln -s ~/.cargo/bin/uv /usr/local/bin/uv

# Copy requirements file
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Install specific momentfm version that works with Python 3.11
RUN uv pip install --system momentfm==0.1.4

# Copy all application files
COPY . .

# Ensure data directory exists
RUN mkdir -p data

# Expose port 5000
EXPOSE 5000

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Run the Starlette app with Uvicorn
CMD ["python", "app.py"]
