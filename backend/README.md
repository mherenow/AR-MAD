# Backend API for AI Image Classifier

FastAPI backend that wraps the existing `classify_image.py` classifier and exposes it via HTTP endpoints.

## Setup

### 1. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Unix/Mac
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

- `ALLOWED_ORIGIN`: CORS allowed origin (default: `http://localhost:5173`)
  - Set this to your frontend URL in production

### Model Checkpoint

The backend requires the model checkpoint to be present at:
```
checkpoints/all_features/checkpoint_epoch_25.pth
```

This path is relative to the project root. Ensure the checkpoint file exists before starting the server, or the application will exit with an error.

## Running the Server

### Option 1: Using the startup script (recommended for quick start)

**Windows:**
```bash
start.bat
```

**Unix/Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

The startup scripts will:
- Check that the virtual environment exists
- Verify the model checkpoint is present
- Set default `ALLOWED_ORIGIN` if not already set
- Start the server on `http://localhost:8000`

### Option 2: Using uvicorn directly

**Development mode (with auto-reload):**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Option 3: Using Python directly

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### `POST /classify`

Classify an uploaded image as AI-generated (FAKE) or real (REAL).

**Request:**
- Method: POST
- Content-Type: `multipart/form-data`
- Body: Single file field named `image`
- Supported formats: JPEG, PNG, BMP, WebP
- Max file size: 10MB

**Response (200 OK):**
```json
{
  "label": "FAKE",
  "confidence": 0.873,
  "prob_fake": 0.873,
  "prob_real": 0.127,
  "logit": 1.234,
  "cam_image_base64": "data:image/png;base64,iVBORw0KG..."
}
```

**Error Responses:**
- `422 Unprocessable Entity`: Invalid file type or corrupted image
- `413 Payload Too Large`: File size exceeds 10MB
- `503 Service Unavailable`: Service at capacity (max 4 concurrent requests)
- `500 Internal Server Error`: Unexpected server error

### `GET /health`

Check if the service is ready to process requests.

**Response (200 OK):**
```json
{
  "status": "ok"
}
```

**Error Response:**
- `503 Service Unavailable`: Model is still loading

## API Documentation

Interactive API documentation is available when the server is running:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Architecture

- **Concurrency Control**: Maximum 4 simultaneous classification requests using asyncio.Semaphore
- **Model Lifecycle**: Model is loaded once at startup and reused for all requests
- **Error Handling**: Structured JSON error responses for all failure cases
- **CORS**: Configured to allow requests from the frontend origin

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_classify.py
```

## Development Notes

- The backend uses the existing `classify_image.py` module without modification
- CAM images are generated using matplotlib with 'Agg' backend (non-interactive)
- All classification logic runs in thread pool to avoid blocking async event loop
- File validation happens both client-side (frontend) and server-side (backend)
