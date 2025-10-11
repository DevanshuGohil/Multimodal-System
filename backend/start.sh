#!/bin/bash
set -e  # Exit on error

# Activate virtual environment
if [ -f "/app/venv/bin/activate" ]; then
    . /app/venv/bin/activate
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Install dependencies if not already installed
if [ ! -d "/app/venv" ]; then
    echo "Setting up virtual environment..."
    python -m venv /app/venv
    . /app/venv/bin/activate
    pip install --upgrade pip
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
fi

# Run database migrations (if any)
# python manage.py migrate

# Start the application
echo "Starting application..."
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
