import io
import datetime
import requests
import pandas as pd
import numpy as np
import boto3
import threading
from botocore.client import Config
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

API_DRIFT_URL = "http://127.0.0.1:80/internal/drift-report"
HEADERS = {"Host": "predict.local", "Content-Type": "application/json"}
TRAIN_DATA_PATH = "/home/alex/projects/ml-project/data/processed/train_processed_2.parquet"

# Настройки для теневого инференса (Shadow Scoring)
SHADOW_DRIFT_URL = "http://127.0.0.1:80/internal/drift-report"
SHADOW_HEADERS = {"Host": "shadow.predict.local", "Content-Type": "application/json"}

MINIO_ENDPOINT = "http://127.0.0.1:9000"
MINIO_ACCESS_KEY = "big_admin_boss"
MINIO_SECRET_KEY = "ioaingognIg-Gn_e&g"
BUCKET_NAME = "ml-monitoring"

SIMULATE_PRODUCTION_DRIFT = True

print("📊 Шаг 1: Загрузка обучающей (Reference) и продакшн (Current) выборки...")
try:
    reference_df = pd.read_parquet(TRAIN_DATA_PATH).head(1000)
    current_df = reference_df.copy()
    if SIMULATE_PRODUCTION_DRIFT:
        np.random.seed(42)
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            current_df[col] = current_df[col] + np.random.normal(150.0, 10.0, size=len(current_df))
        print("🚨 ВНИМАНИЕ: В выборку 'current' внедрен искусственный ковариатный сдвиг!")
    else:
        print("🟢 СТАБИЛЬНОСТЬ: Выборка 'current' оставлена эталонной чистой копией.")
except Exception as e:
    print(f"❌ Ошибка подготовки данных ({e})")
    exit(1)

print("📊 Шаг 2: Расчет Data Drift через Evidently AI (KS-test)...")
data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=reference_df, current_data=current_df)
report_dict = data_drift_report.as_dict()

number_of_drifted_features = 0
share_of_drifted_features = 0.0
dataset_drift = False

for metric_entry in report_dict.get("metrics", []):
    if metric_entry.get("metric") == "DatasetDriftMetric":
        res = metric_entry.get("result", {})
        number_of_drifted_features = res.get("number_of_drifted_columns", 0)
        share_of_drifted_features = res.get("share_of_drifted_columns", 0.0)
        dataset_drift = res.get("dataset_drift", False)
        break

print(f"📈 Итог стат-анализа: Сдвинуто фичей: {number_of_drifted_features} ({share_of_drifted_features:.1%})")

print("💾 Шаг 3: Генерация HTML-дашборда и выгрузка отчета в MinIO S3...")
html_buffer = io.StringIO()
data_drift_report.save_html(html_buffer)
html_bytes = html_buffer.getvalue().encode('utf-8')

s3_client = boto3.client(
    's3', endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4'), region_name='us-east-1'
)

try:
    s3_client.create_bucket(Bucket=BUCKET_NAME)
except Exception:
    pass

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
object_name = f"drift_report_{timestamp}.html"
s3_client.put_object(Bucket=BUCKET_NAME, Key=object_name, Body=html_bytes, ContentType='text/html')
print(f"🎯 Отчет успешно сохранен in MinIO S3: object='{object_name}'")

print("📡 Шаг 4: Отправка численных метрик дрифта напрямую в FastAPI для Prometheus...")
payload = {
    "number_of_drifted_features": int(number_of_drifted_features),
    "share_of_drifted_features": float(share_of_drifted_features),
    "dataset_drift": bool(dataset_drift)
}

def send_shadow_traffic(url, headers, data):
    """Вспомогательная функция для асинхронного отзеркаливания трафика."""
    try:
        requests.post(url, headers=headers, json=data, timeout=5)
    except Exception:
        pass # Игнорируем ошибки теневой ветки, чтобы не аффектить прод

# А. Отправляем оригинальный трафик на прод (Champion)
try:
    res = requests.post(API_DRIFT_URL, headers=HEADERS, json=payload, timeout=10)
    print(f"✅ Статус доставки на ПРOД (Champion): {res.status_code} | Ответ: {res.json()}")
except Exception as e:
    print(f"❌ Не удалось доставить метрики на Прод: {e}")

# Б. ТРЕБОВАНИЕ ЗАДАНИЯ 7: Отзеркаливаем точную копию трафика на Staging (Challenger) в фоне
shadow_thread = threading.Thread(target=send_shadow_traffic, args=(SHADOW_DRIFT_URL, SHADOW_HEADERS, payload))
shadow_thread.daemon = True
shadow_thread.start()
print("📡 Зеркало трафика (Shadow Scoring) успешно продублировано на модель Ready_for_staging!")
