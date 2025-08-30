# PS-05 Challenge - Complete Backend Solution
# Optimized for A100 GPU with 2-hour evaluation time limit

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV MPLCONFIGDIR=/tmp
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    libc6 \
    libtesseract-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-urd \
    tesseract-ocr-ara \
    tesseract-ocr-nep \
    tesseract-ocr-fas \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.minimal.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.minimal.txt

# Optional GPU/ML heavy dependencies (keeps existing requirements intact)
# Build with: docker build --build-arg INSTALL_GPU_DEPS=1 -t ps05-backend:gpu .
ARG INSTALL_GPU_DEPS=0
RUN if [ "$INSTALL_GPU_DEPS" = "1" ]; then \
    echo "Installing CUDA-enabled PyTorch and ML deps..." && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio && \
    pip install --no-cache-dir \
        opencv-python-headless \
        ultralytics \
        transformers \
        timm \
        safetensors \
        sentencepiece \
        easyocr \
        fasttext \
        onnxruntime-gpu \
        pycocotools && \
    curl -L -o /app/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin; \
    else echo "Skipping GPU deps installation" ; fi

# Copy application code
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY ps05.py ./
# Keep only the main entrypoints
COPY start_backend.py ./
## Include pre-fetched models in the image if present
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p datasets results logs models data/api_datasets data/api_results

# Set permissions
RUN chmod +x ps05.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run uvicorn directly (no ps05.py args)
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
