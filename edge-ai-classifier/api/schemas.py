"""
Pydantic schemas for the Edge AI Classifier API.
"""
from typing import List, Any
from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str = Field(..., description="Human-readable class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0, 1]")

    model_config = {"json_schema_extra": {"example": {"label": "golden retriever", "confidence": 0.921}}}


class ClassificationResponse(BaseModel):
    predictions: List[Prediction] = Field(..., description="Top-K predictions, sorted by confidence descending")
    inference_time_ms: float = Field(..., description="Total preprocessing + inference time in milliseconds")
    model_version: str = Field(..., description="Version string of the deployed model")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {"label": "golden retriever", "confidence": 0.921},
                    {"label": "Labrador retriever", "confidence": 0.043},
                ],
                "inference_time_ms": 11.8,
                "model_version": "mobilenetv2-int8-v1.2",
            }
        }
    }


class BatchClassificationResponse(BaseModel):
    results: List[Any] = Field(..., description="Per-image classification results (or error object)")
    total_images: int = Field(..., description="Number of images in the batch")
    total_inference_time_ms: float = Field(..., description="Wall-clock time for the entire batch")


class ModelInfo(BaseModel):
    name: str
    version: str
    size_mb: float
    classes: int
    input_shape: List[int]


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' or 'degraded'")
    model: ModelInfo
    uptime_seconds: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model": {
                    "name": "mobilenetv2-int8",
                    "version": "1.2.0",
                    "size_mb": 3.4,
                    "classes": 1000,
                    "input_shape": [1, 224, 224, 3],
                },
                "uptime_seconds": 3842,
            }
        }
    }
