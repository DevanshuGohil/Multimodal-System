#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run database migrations (if any)
# python manage.py migrate

# Start the application
uvicorn main:app --host 0.0.0.0 --port $PORT
