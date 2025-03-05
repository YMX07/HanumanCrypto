gunicorn app:app -b 0.0.0.0:$PORT
worker: celery -A app.celery worker --loglevel=info
