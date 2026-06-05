"""FastAPI backend for AI image classification.

This module provides HTTP API endpoints that wrap the existing classify_image.py
classifier, enabling web-based image classification with CAM visualization.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel, Field
import asyncio
import os
import sys
import logging
import io
import base64

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environment
import matplotlib.pyplot as plt

from PIL import Image
import torch
from torchvision import transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Image Classifier API", version="1.0.0")

# Initialize app state
app.state.model = None
app.state.model_ready = False
app.state.semaphore = asyncio.Semaphore(4)  # Max 4 concurrent requests

# CORS configuration
allowed_origin = os.getenv("ALLOWED_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

logger.info(f"CORS configured for origin: {allowed_origin}")


from pydantic import ConfigDict

class ClassificationResult(BaseModel):
    label: str = Field(..., pattern="^(FAKE|REAL)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    prob_fake: float = Field(..., ge=0.0, le=1.0)
    prob_real: float = Field(..., ge=0.0, le=1.0)
    logit: float
    cam_image_base64: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": "FAKE",
                "confidence": 0.873,
                "prob_fake": 0.873,
                "prob_real": 0.127,
                "logit": 1.234,
                "cam_image_base64": "data:image/png;base64,..."
            }
        }
    )

@app.on_event("startup")
async def startup_event():
    """Load model once at startup."""

    # Project root = AR-MAD
    BASE_DIR = Path(__file__).resolve().parent.parent

    checkpoint_path = (
    BASE_DIR
    / "checkpoints"
    / "checkpoint_epoch_25.pth"
)

    logger.info(f"Looking for checkpoint at: {checkpoint_path}")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Loading model from {checkpoint_path}...")

    try:
        # Add ai-image-detector to Python path
        detector_path = BASE_DIR / "ai-image-detector"

        if str(detector_path) not in sys.path:
            sys.path.insert(0, str(detector_path))

        from classify_image import load_model

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Using device: {device}")

        loop = asyncio.get_event_loop()

        app.state.model = await loop.run_in_executor(
            None,
            load_model,
            str(checkpoint_path),
            device,
        )

        app.state.model_ready = True

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        sys.exit(1)


@app.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        dict: Status object with "ok" when model is ready, or status field when loading
        
    Raises:
        HTTPException: 503 when model is still loading
    """
    if not app.state.model_ready:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "detail": "Model is still loading"}
        )
    return {"status": "ok"}


@app.post("/classify", response_model=ClassificationResult)
async def classify(image: UploadFile = File(...)):
    """Classify an uploaded image as FAKE or REAL.
    
    Args:
        image: Uploaded image file (JPEG, PNG, BMP, or WebP)
        
    Returns:
        ClassificationResult: Label, confidence, probabilities, and CAM heatmap
        
    Raises:
        HTTPException: 422 for unsupported file type or invalid image
        HTTPException: 413 for file size exceeding 10MB
        HTTPException: 503 for service at capacity or model not ready
        HTTPException: 500 for internal server errors
    """
    if not app.state.model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    # Validate file type
    supported_types = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
    if image.content_type not in supported_types:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {image.content_type}. Supported formats: JPEG, PNG, BMP, WebP"
        )
    
    # Read and validate file size
    contents = await image.read()
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Empty file")
    if len(contents) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=413,
            detail="File size exceeds 10MB limit"
        )
    
    # Check concurrency limit - reject immediately if at capacity
    if app.state.semaphore.locked():
        logger.warning("Semaphore at capacity, rejecting request")
        raise HTTPException(
            status_code=503,
            detail="Service at capacity. Maximum 4 concurrent requests allowed."
        )
    
    # Process classification with concurrency control
    async with app.state.semaphore:
        try:
            logger.info(f"Processing image: {image.filename}, size: {len(contents)} bytes")
            
            # Run blocking classification in thread pool executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, classify_image_sync, contents)
            
            logger.info(f"Classification result: label={result.label}, confidence={result.confidence:.4f}")
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Classification failed")
            raise HTTPException(
                status_code=500,
                detail="Internal server error during classification"
            )


def classify_image_sync(image_bytes: bytes) -> ClassificationResult:
    """Synchronous classification wrapper (runs in thread pool executor).

    Loads image from raw bytes, runs EffectiveWeightCAM inference, generates
    the same three-panel CAM figure as classify_image.py's visualize(), and
    returns a fully-populated ClassificationResult.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        ClassificationResult with label, confidence, probabilities, logit,
        and base64-encoded three-panel CAM figure PNG.

    Raises:
        HTTPException: 422 if image bytes cannot be decoded as a valid image.
    """
    # Add ai-image-detector to the module search path so we can import from it
    detector_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai-image-detector')
    if detector_path not in sys.path:
        sys.path.insert(0, detector_path)

    from classify_image import EffectiveWeightCAM, classify_image, FAKE_THRESHOLD

    # 1. Load image from bytes using PIL
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot decode image data: {e}"
        )

    # 2. Transform image to tensor (resize to 256×256, normalize with ImageNet stats)
    #    Matches load_image() in classify_image.py exactly.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img_resized = img.resize((256, 256))  # PIL image for visualization
    tensor = transform(img).unsqueeze(0)  # (1, 3, 256, 256)

    # Move tensor to the same device as the model
    model = app.state.model
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    # 3. Instantiate EffectiveWeightCAM and run inference
    cam_engine = EffectiveWeightCAM(model)
    try:
        cam, logit, p_fake = cam_engine.generate_cam(tensor)
    finally:
        # Always remove hooks to avoid memory leaks and stale state
        cam_engine.remove_hooks()

    # 4. Classify using the same calibrated threshold as classify_image.py
    #    (FAKE_THRESHOLD = 0.9244; confidence = prob_fake if FAKE else prob_real)
    prediction, confidence, prob_real, prob_fake = classify_image(logit)
    label = "FAKE" if prediction == 1 else "REAL"

    # 5. Render overlay-only panel (image + masked CAM heatmap)
    img_np = np.array(img_resized)
    h, w   = img_np.shape[:2]

    # Upsample CAM to match display image size (same as visualize())
    cam_up = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_resized)
    masked_cam = np.ma.masked_where(cam_up < 0.05, cam_up)
    ax.imshow(masked_cam, cmap='hot', alpha=0.65, vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)

    # Render figure to in-memory PNG buffer (no disk I/O)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    cam_image_base64 = f"data:image/png;base64,{img_b64}"

    return ClassificationResult(
        label=label,
        confidence=float(confidence),
        prob_fake=float(prob_fake),
        prob_real=float(prob_real),
        logit=float(logit),
        cam_image_base64=cam_image_base64,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
