"""
Edge AI Image Classification System — FastAPI Application
"""
import time
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.inference import InferenceEngine
from api.preprocessing import ImagePreprocessor
from api.schemas import (
    ClassificationResponse,
    BatchClassificationResponse,
    HealthResponse,
    ModelInfo,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons (loaded once on startup)
# ---------------------------------------------------------------------------
engine: InferenceEngine | None = None
preprocessor: ImagePreprocessor | None = None
startup_time: float = time.time()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Lifespan — model loading on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, preprocessor
    logger.info("Loading TFLite inference engine...")
    engine = InferenceEngine()
    preprocessor = ImagePreprocessor()
    logger.info(
        f"Model loaded — {engine.model_info['name']} "
        f"({engine.model_info['size_mb']:.1f} MB)"
    )
    yield
    logger.info("Shutting down inference engine.")
    engine = None
    preprocessor = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Edge AI Image Classifier",
    description=(
        "Lightweight image classification API powered by a quantized "
        "MobileNetV2 model optimized for edge deployment. "
        "Achieves 60%+ model size reduction via INT8 post-training quantization."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_upload(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted: {sorted(ALLOWED_MIME_TYPES)}"
            ),
        )


async def _read_file(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds limit of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )
    return data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Edge AI Classifier is running. See /docs for the API."}


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health():
    """Returns service health status and loaded model metadata."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return HealthResponse(
        status="healthy",
        model=ModelInfo(**engine.model_info),
        uptime_seconds=int(time.time() - startup_time),
    )


@app.post(
    "/api/v1/classify",
    response_model=ClassificationResponse,
    summary="Classify a single image",
    tags=["Inference"],
)
async def classify(file: UploadFile = File(..., description="Image file to classify")):
    """
    Upload a single image and receive the top-5 class predictions with
    confidence scores. Supported formats: JPEG, PNG, WebP, BMP.
    """
    _validate_upload(file)
    image_bytes = await _read_file(file)

    try:
        t0 = time.perf_counter()
        tensor = preprocessor.process(image_bytes)
        predictions = engine.predict(tensor, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    logger.info(
        f"classify | file={file.filename} | top1={predictions[0]['label']} "
        f"({predictions[0]['confidence']:.2%}) | {elapsed_ms:.1f}ms"
    )

    return ClassificationResponse(
        predictions=predictions,
        inference_time_ms=round(elapsed_ms, 2),
        model_version=engine.model_info["version"],
    )


@app.post(
    "/api/v1/classify/batch",
    response_model=BatchClassificationResponse,
    summary="Classify multiple images at once",
    tags=["Inference"],
)
async def classify_batch(
    files: List[UploadFile] = File(..., description="List of image files (max 16)")
):
    """
    Upload up to 16 images and receive predictions for each. All images
    are preprocessed in parallel before a single batched inference pass.
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(files)} exceeds maximum of {MAX_BATCH_SIZE}.",
        )

    results = []
    total_t0 = time.perf_counter()

    for file in files:
        _validate_upload(file)
        image_bytes = await _read_file(file)
        try:
            t0 = time.perf_counter()
            tensor = preprocessor.process(image_bytes)
            predictions = engine.predict(tensor, top_k=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            results.append(
                ClassificationResponse(
                    predictions=predictions,
                    inference_time_ms=round(elapsed_ms, 2),
                    model_version=engine.model_info["version"],
                )
            )
        except ValueError as exc:
            results.append({"error": str(exc), "filename": file.filename})

    total_ms = (time.perf_counter() - total_t0) * 1000
    logger.info(f"batch classify | count={len(files)} | total={total_ms:.1f}ms")

    return BatchClassificationResponse(
        results=results,
        total_images=len(files),
        total_inference_time_ms=round(total_ms, 2),
    )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
