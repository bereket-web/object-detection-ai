from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from boardwatch.constants import CHECKPOINT_PATH, CLASS_NAMES, DEFAULT_SCORE_THRESHOLD
from boardwatch.inference import BoardDetector

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
STATIC_ROOT = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="BoardWatch Object Detection",
    description="Local CPU detector for tic-tac-toe cell occupancy.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_detector() -> BoardDetector:
    return BoardDetector(CHECKPOINT_PATH)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": CHECKPOINT_PATH.is_file(),
        "checkpoint": CHECKPOINT_PATH.name if CHECKPOINT_PATH.is_file() else None,
        "classes": list(CLASS_NAMES),
        "device": "cpu",
    }


@app.post("/detect")
def detect(
    file: Annotated[UploadFile, File(...)], score_threshold: float = DEFAULT_SCORE_THRESHOLD
) -> dict[str, object]:
    if not 0.0 <= score_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="score_threshold must be in [0, 1]")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit")
    try:
        with Image.open(BytesIO(content)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image exceeds the 24 megapixel limit")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="Upload must be a readable JPEG, PNG, or WebP image"
        ) from exc

    try:
        detector = get_detector()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Local model not found. Run `boardwatch prepare` and `boardwatch train` first.",
        ) from exc

    detections = detector.predict(image, score_threshold=score_threshold)
    return {
        "width": image.width,
        "height": image.height,
        "score_threshold": score_threshold,
        "count": len(detections),
        "detections": detections,
    }
