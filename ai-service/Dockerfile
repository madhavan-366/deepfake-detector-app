# Use Python 3.9 Slim
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# --- FIX: INSTALL BUILD TOOLS (GCC) ---
# This allows Alibi/Spacy/Blis to compile
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the extra AI libraries (The Giant Line)
RUN pip install opencv-python-headless facenet-pytorch alibi matplotlib scikit-image pillow requests flask-cors

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 5000

RUN pip install grad-cam lime timm

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
# Start the app
CMD ["python", "app.py"]