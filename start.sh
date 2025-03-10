#!/bin/bash

# Exit on error
set -e

echo "Starting services..."

# Start Celery worker in the background
echo "Starting Celery worker..."
celery -A app.celery worker --loglevel=info &
CELERY_PID=$!

# Log Celery worker PID
echo "Celery worker started with PID: $CELERY_PID"

# Start Gunicorn in the foreground
echo "Starting Gunicorn..."
gunicorn --timeout 120 -w 4 app:app --bind 0.0.0.0:10000

# Note: This part will only execute if Gunicorn exits
# Kill the Celery worker if Gunicorn stops
if ps -p $CELERY_PID > /dev/null; then
    echo "Stopping Celery worker..."
    kill $CELERY_PID
fi