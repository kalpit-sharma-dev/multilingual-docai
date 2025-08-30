# PS-05 Document Understanding Backend Dockerfile
# Optimized for Ubuntu 24.04 with GPU support

FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-pip \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python3
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for document processing
RUN pip3 install --no-cache-dir \
    opencv-python-headless \
    pillow \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scikit-image \
    tqdm \
    PyYAML \
    requests \
    aiofiles \
    python-multipart

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/api_datasets data/api_results models

# Set permissions
RUN chmod +x scripts/*.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/status || exit 1

# Default command
CMD ["python3", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
