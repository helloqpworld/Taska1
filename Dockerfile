FROM python:3.12.3

RUN pip install mlflow==3.1.0 boto3 psycopg2

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB} \
    --host 0.0.0.0 \
    --port 5000 \
    --artifacts-destination s3://${MINIO_BUCKET_NAME}/artifacts \
    --serve-artifacts \
    --gunicorn-opts "--timeout 180 --workers 2"
