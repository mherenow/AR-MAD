#!/bin/bash
# Backend startup script for Unix/Linux/Mac
# This script starts the FastAPI backend server with uvicorn

# Exit on error
set -e

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found at ./venv"
    echo "Please run setup first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if checkpoint exists
CHECKPOINT_PATH="../checkpoints/all_features/checkpoint_epoch_25.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Model checkpoint not found at $CHECKPOINT_PATH"
    echo "Please ensure the checkpoint file is present before starting the server."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Set default ALLOWED_ORIGIN if not set
if [ -z "$ALLOWED_ORIGIN" ]; then
    export ALLOWED_ORIGIN="http://localhost:5173"
    echo "Using default ALLOWED_ORIGIN: $ALLOWED_ORIGIN"
fi

# Start server
echo "Starting FastAPI backend server..."
echo "API will be available at: http://localhost:8000"
echo "API docs available at: http://localhost:8000/docs"
echo ""
uvicorn main:app --host 0.0.0.0 --port 8000
