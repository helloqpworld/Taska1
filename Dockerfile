FROM python:3.12.3

RUN pip install mlflow==3.1.0 boto3 psycopg2

# DevOps-практика: объявляем порты и команду запуска прямо внутри образа
EXPOSE 5000

CMD mlflow server \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB} \
    --host 0.0.0.0 \
    --serve-artifacts \
    --artifacts-destination s3://${MINIO_BUCKET_NAME}/artifacts
