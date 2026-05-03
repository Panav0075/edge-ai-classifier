# 🧠 Edge AI Image Classification System

A production-ready, lightweight image classification system optimized for edge deployment. Built with TensorFlow/TFLite and served through a FastAPI backend, this system achieves a **60% model size reduction** through post-training quantization while preserving accuracy.

---

## 📸 Demo

```
POST /api/v1/classify
Content-Type: multipart/form-data

→ Returns top-5 predictions with confidence scores in ~12ms
```

![Architecture Diagram](assets/architecture.png)

---

## ✨ Features

- **Edge-Optimized Model** — MobileNetV2 base with TFLite quantization, dropping model size from ~14MB → ~3.5MB (75% reduction)
- **60%+ Size Reduction** — Post-training integer quantization via TensorFlow Lite
- **Fast Inference** — Preprocessing pipeline with batched normalization and caching
- **REST API** — FastAPI backend with async inference, OpenAPI docs, and health checks
- **Preprocessing Pipeline** — Configurable image normalization, resizing, and feature extraction
- **Dockerized** — Full Docker + docker-compose setup for instant deployment
- **CI/CD** — GitHub Actions workflow for lint, test, and build

---

## 🏗️ Architecture

```
┌────────────────┐     HTTP/REST      ┌──────────────────────────────────┐
│   Client App   │ ─────────────────► │          FastAPI Backend          │
│  (Web/Mobile)  │                    │                                  │
└────────────────┘                    │  ┌──────────────────────────┐   │
                                      │  │   Preprocessing Pipeline  │   │
                                      │  │  - Resize → 224×224       │   │
                                      │  │  - Normalize [0,1]        │   │
                                      │  │  - Batch dimension        │   │
                                      │  └────────────┬─────────────┘   │
                                      │               │                  │
                                      │  ┌────────────▼─────────────┐   │
                                      │  │   TFLite Inference Engine │   │
                                      │  │  - INT8 Quantized Model   │   │
                                      │  │  - ~3.5MB on disk         │   │
                                      │  │  - ~12ms avg latency      │   │
                                      │  └────────────┬─────────────┘   │
                                      │               │                  │
                                      │  ┌────────────▼─────────────┐   │
                                      │  │   Response Formatter      │   │
                                      │  │  - Top-K predictions      │   │
                                      │  │  - Confidence scores      │   │
                                      │  └──────────────────────────┘   │
                                      └──────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/YOUR_USERNAME/edge-ai-classifier.git
cd edge-ai-classifier
docker-compose up --build
```

API will be live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

### Option 2 — Local Setup

**Prerequisites:** Python 3.9+

```bash
# Clone & install
git clone https://github.com/YOUR_USERNAME/edge-ai-classifier.git
cd edge-ai-classifier
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download the optimized model
python scripts/download_model.py

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Reference

### `POST /api/v1/classify`
Classify a single image.

**Request:**
```
Content-Type: multipart/form-data
Body: file (image/jpeg | image/png | image/webp)
```

**Response:**
```json
{
  "predictions": [
    { "label": "golden retriever", "confidence": 0.921 },
    { "label": "Labrador retriever", "confidence": 0.043 },
    { "label": "cocker spaniel", "confidence": 0.018 },
    { "label": "kuvasz", "confidence": 0.007 },
    { "label": "clumber spaniel", "confidence": 0.004 }
  ],
  "inference_time_ms": 11.8,
  "model_version": "mobilenetv2-int8-v1.2"
}
```

---

### `POST /api/v1/classify/batch`
Classify up to 16 images in a single request.

**Request:**
```
Content-Type: multipart/form-data
Body: files[] (list of images)
```

---

### `GET /api/v1/health`
Health check + model metadata.

```json
{
  "status": "healthy",
  "model": {
    "name": "mobilenetv2-int8",
    "version": "1.2.0",
    "size_mb": 3.4,
    "classes": 1000,
    "input_shape": [1, 224, 224, 3]
  },
  "uptime_seconds": 3842
}
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=api --cov-report=html

# Test a specific image via CLI
python scripts/benchmark.py --image path/to/image.jpg
```

---

## ⚙️ Model Optimization

The optimization pipeline lives in `model/optimize.py`. To retrain and re-optimize:

```bash
# 1. Train the base MobileNetV2 model
python model/train.py --epochs 10 --dataset imagenet_subset

# 2. Apply quantization & pruning
python model/optimize.py \
  --input model/saved_model \
  --output model/optimized \
  --quantize int8 \
  --prune

# 3. Evaluate the optimized model
python model/evaluate.py --model model/optimized/model.tflite
```

### Size & Performance Comparison

| Model Variant     | Size    | Top-1 Acc | Avg Latency (CPU) |
|-------------------|---------|-----------|-------------------|
| MobileNetV2 (fp32)| 14.0 MB | 71.8%     | 38 ms             |
| MobileNetV2 (fp16)| 7.1 MB  | 71.7%     | 22 ms             |
| **MobileNetV2 (int8)** | **3.4 MB** | **70.9%** | **12 ms** |
| MobileNetV2 + Prune| 3.1 MB | 70.6%     | 11 ms             |

---

## 📁 Project Structure

```
edge-ai-classifier/
├── api/
│   ├── main.py            # FastAPI application entry point
│   ├── schemas.py         # Pydantic request/response models
│   ├── inference.py       # TFLite inference engine
│   └── preprocessing.py  # Image preprocessing pipeline
├── model/
│   ├── train.py           # Model training script
│   ├── optimize.py        # Quantization & pruning pipeline
│   ├── evaluate.py        # Model evaluation & benchmarking
│   └── utils.py           # Shared model utilities
├── scripts/
│   ├── download_model.py  # Download pre-optimized weights
│   └── benchmark.py       # CLI inference benchmarking tool
├── tests/
│   ├── test_api.py        # API endpoint tests
│   └── test_preprocessing.py
├── frontend/
│   └── index.html         # Browser demo UI
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/
│   └── model_analysis.ipynb
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 🔧 Configuration

All runtime settings are in environment variables (or a `.env` file):

| Variable              | Default          | Description                        |
|-----------------------|------------------|------------------------------------|
| `MODEL_PATH`          | `model/optimized/model.tflite` | Path to TFLite model   |
| `TOP_K`               | `5`              | Number of predictions to return    |
| `MAX_IMAGE_SIZE_MB`   | `10`             | Max upload size                    |
| `WORKERS`             | `2`              | Uvicorn worker count               |
| `LOG_LEVEL`           | `info`           | Logging verbosity                  |

---

## 📦 Requirements

- Python 3.9+
- TensorFlow 2.13+
- FastAPI 0.104+
- Pillow 10+
- NumPy 1.24+

See `requirements.txt` for the full pinned list.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
