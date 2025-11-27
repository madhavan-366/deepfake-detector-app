# Use Python 3.9 Slim
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install basic tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- OPTIMIZATION: Upgrade pip so it can find the "Fast Wheels" ---
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# --- OPTIMIZATION: Prefer Binary (Pre-cooked) versions ---
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Install the extra AI libraries (Force binary for speed)
RUN pip install --no-cache-dir --prefer-binary opencv-python-headless facenet-pytorch alibi matplotlib scikit-image pillow requests flask-cors

# Copy the rest of the code
COPY . .

# Install the final layer (Force binary)
RUN pip install --no-cache-dir --prefer-binary grad-cam lime timm

# Expose port
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]