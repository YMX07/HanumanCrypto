web: gunicorn app:app --bind 0.0.0.0:10000
worker: celery -A app.celery worker --loglevel=info
