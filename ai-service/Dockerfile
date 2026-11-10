# Use Python 3.9 Slim
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and build tools
RUN pip install --upgrade pip setuptools wheel

# 3. Install CPU-only PyTorch and Torchvision (reduces container size and prevents CUDA memory overhead)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# 4. Install AI Libraries + Flask + Alibi (numpy < 2.0 pinned for Alibi compatibility)
RUN pip install --no-cache-dir --prefer-binary \
    "numpy<2.0" \
    flask \
    flask-cors \
    opencv-python-headless \
    facenet-pytorch \
    matplotlib \
    scikit-image \
    pillow \
    requests \
    grad-cam \
    lime \
    timm \
    alibi

# 5. Copy application code and model weights
COPY . .

# 6. Set environment variables
ENV PORT=7860 \
    TORCH_HOME=/tmp/torch

# 7. Expose Hugging Face Space default port
EXPOSE 7860

# 8. Start application
CMD ["python", "app.py"]
