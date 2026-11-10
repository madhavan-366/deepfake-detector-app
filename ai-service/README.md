---
title: Deepfake Detector AI Service
emoji: 🎭
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Deepfake Detector - AI Service Engine

Microservice powering the Deepfake Detector application using PyTorch, XceptionNet, and explainability frameworks.

## Features

- **XceptionNet Face Deepfake Detection**: Binary classification (Real vs. Fake).
- **Face Localization**: MTCNN bounding box detection and alignment.
- **Explainability**:
  - Grad-CAM heatmap visualization (Novice Mode).
  - LIME perturbation analysis (Expert Mode).
  - Counterfactual explanation generation.
  - Stress testing (Gaussian noise, motion blur, JPEG compression).

## API Endpoints

- `GET /` - Health check and service metadata
- `GET /health` - Health status confirmation
- `POST /api/detect` - Face deepfake prediction and heatmap generation (multipart/form-data with `image` and `mode`)
- `POST /api/stress-test` - Robustness testing against simulated corruption
- `POST /api/counterfactual` - Minimal perturbation counterfactual analysis
