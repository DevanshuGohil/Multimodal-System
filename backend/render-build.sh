#!/bin/bash
# Exit on error
set -o errexit

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Run the application
exec uvicorn main:app --host 0.0.0.0 --port $PORT
