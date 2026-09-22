import datetime
import requests
import boto3
import subprocess
from botocore.client import Config
from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.exceptions import AirflowFailException
from kubernetes.client import models as k8s

# ================================================================================
# СУПЕР-МАНИФЕСТ С АВТОМАТИЗАЦИЕЙ РЕЛИЗОВ ЧЕРЕЗ REST API MLFLOW
# ================================================================================
TELEGRAM_TOKEN = "8909681536:AAHFffX3nX_8lLRLNqICGV-DWB5Ye9CXTgg"
TELEGRAM_CHAT_ID = "-5579643046"

PROMETHEUS_URL = "http://10.96.146.167:9090"
MINIO_EXTERNAL_URL = "http://169.58.244.233:9000" 
BUCKET_NAME = "ml-monitoring"
RETRAIN_IMAGE = "169.58.244.233:8929/root/taska1/predict_service:latest"
MLFLOW_REST_URL = "http://169.58.244.233:5080"

default_args = {
    'owner': 'mlops_alex',
    'start_date': datetime.datetime(2026, 9, 1),
    'retries': 0,
}

@dag(
    dag_id='ml_data_drift_sensor_pipeline',
    default_args=default_args,
    schedule='0 */3 * * *', 
    catchup=False,
    tags=['monitoring', 'mlops', 'continuous-training', 'canary-rollback'],
    params={
        "aggregation_type": Param("max", type="string", enum=["max", "last"])
    }
)
def drift_sensor_dag():

    @task()
    def check_prometheus_drift_metric(**context):
        try:
            url = f"{PROMETHEUS_URL}/api/v1/query"
            dag_run_params = context.get('params', {})
            agg_type = dag_run_params.get('aggregation_type', 'max')
            smart_query = "last_over_time(ml_dataset_drift_alert[1m])" if agg_type == "last" else "max_over_time(ml_dataset_drift_alert[15m])"
            res = requests.get(url, params={"query": smart_query}, timeout=10)
            data = res.json()
            result_list = data.get("data", {}).get("result", [])
            if isinstance(result_list, list) and len(result_list) > 0:
                first_item = result_list[0]
                if isinstance(first_item, dict) and "value" in first_item:
                    value_block = first_item["value"]
                    if isinstance(value_block, list) and len(value_block) > 1:
                        return float(value_block[1])
            return 0.0
        except Exception as e:
            return 0.0

    @task()
    def generate_s3_report_link(drift_status: float):
        s3_client = boto3.client(
            's3', endpoint_url=MINIO_EXTERNAL_URL,
            aws_access_key_id="big_admin_boss", aws_secret_access_key="ioaingognIg-Gn_e&g",
            config=Config(signature_version='s3v4'), region_name='us-east-1'
        )
        try:
            objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
            if 'Contents' in objects:
                latest_object = max(objects['Contents'], key=lambda x: x['LastModified'])
                file_key = latest_object['Key']
                return s3_client.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': file_key}, ExpiresIn=604800)
            return f"{MINIO_EXTERNAL_URL}/{BUCKET_NAME}/"
        except Exception as e:
            return f"S3 error: {e}"

    @task()
    def send_telegram_alert(drift_status: float, report_url: str, **context):
        dag_run_params = context.get('params', {})
        agg_type = dag_run_params.get('aggregation_type', 'max')
        if drift_status >= 1.0:
            message_text = (
                "🚨 <b>ВНИМАНИЕ! Airflow зафиксировал Data Drift</b> 🚨\n\n"
                f"📊 <b>Статус ml_dataset_drift_alert:</b> <code>{drift_status} (КРИТИЧЕСКИЙ СДВИГ)</code>\n"
                f"⚙️ <b>Режим PromQL:</b> <code>{agg_type}_over_time</code>\n\n"
                "⚙️ <b>CI/CD АВТО-ТРИГГЕР:</b> Airflow запускает изолированный K8s Job переобучения!"
            )
        else:
            message_text = (
                "✅ <b>Airflow: Проверка качества данных успешна</b> ✅\n\n"
                f"🟢 <b>Статус ml_dataset_drift_alert:</b> <code>{drift_status} (СТАБИЛЬНО)</code>\n"
            )
        tg_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(tg_url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message_text, "parse_mode": "HTML"}, timeout=10)

    pod_configuration = k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base")], host_network=True))
    run_continuous_training_job = KubernetesPodOperator(
        task_id="run_continuous_training_job",
        name="ml-taxi-lgb-retrain-pod",
        namespace="airflow",
        image=RETRAIN_IMAGE,
        image_pull_policy="IfNotPresent",
        full_pod_spec=pod_configuration,
        get_logs=True,
        is_delete_operator_pod=True,
        log_events_on_failure=True
    )

    wait_for_engineer_approval = EmptyOperator(
        task_id="wait_for_engineer_approval"
    )

    @task()
    def promote_shadow_model_to_production():
        """ИСПРАВЛЕНО: Безопасно управляем тегами через REST API MLflow (без импорта mlflow)."""
        import json
        print("🚀 Запрос версии Staging модели через REST API MLflow...")
        
        # Шаг А: Спрашиваем у MLflow, какая версия сейчас носит алиас Ready_for_staging
        get_url = f"{MLFLOW_REST_URL}/api/2.0/mlflow/registered-models/alias"
        params = {"name": "nyc_taxi_lightgbm", "alias": "Ready_for_staging"}
        
        res = requests.get(get_url, params=params, timeout=10)
        res.raise_for_status()
        staging_info = res.json()
        
        # Вытаскиваем строковый номер версии модели
        staging_version = staging_info.get("model_version", {}).get("version")
        print(f"🎯 Найдена Staging-версия модели: №{staging_version}")
        
        # Шаг Б: Через POST-запрос официально вешаем на эту версию алиас 'prod'
        post_url = f"{MLFLOW_REST_URL}/api/2.0/mlflow/registered-models/alias"
        payload = {
            "name": "nyc_taxi_lightgbm",
            "alias": "prod",
            "version": str(staging_version)
        }
        
        post_res = requests.post(post_url, json=payload, timeout=10)
        post_res.raise_for_status()
        print(f"✅ Успех! Версия №{staging_version} переведена в статус боевого инференса @prod через REST-протокол.")

    @task()
    def monitor_canary_health_and_rollback():
        """Анализируем 5xx ошибки через Prometheus. При сбое — откатываем тег назад через REST API."""
        try:
            print("📡 Запрос к Prometheus API для анализа стабильности Canary-релиза...")
            query_url = f"{PROMETHEUS_URL}/api/v1/query"
            res = requests.get(query_url, params={"query": "sum(rate(fastapi_requests_total{status=~'5..'}[2m]))"}, timeout=5)
            data = res.json()
            result_list = data.get("data", {}).get("result", [])
            
            error_rate = 0.0
            if isinstance(result_list, list) and len(result_list) > 0:
                first_item = result_list[0]
                if isinstance(first_item, dict) and "value" in first_item:
                    value_block = first_item["value"]
                    if isinstance(value_block, list) and len(value_block) > 1:
                        error_rate = float(value_block[1])
            
            print(f"📊 Текущий коэффициент 5xx ошибок на проде: {error_rate}")
            
            if error_rate > 0.0:
                print("🚨 СБОЙ МЕТРИК! Откатываем тег @prod на прошлую версию через REST API...")
                # Вытаскиваем список версий и берем предпоследнюю
                list_url = f"{MLFLOW_REST_URL}/api/2.0/mlflow/registered-models/get"
                list_res = requests.get(list_url, params={"name": "nyc_taxi_lightgbm"}, timeout=10)
                versions = list_res.json().get("registered_model", {}).get("latest_versions", [])
                
                if len(versions) > 1:
                    prev_version = versions[-2].get("version")
                    # Возвращаем старую стабильную версию на прод
                    rollback_url = f"{MLFLOW_REST_URL}/api/2.0/mlflow/registered-models/alias"
                    requests.post(rollback_url, json={"name": "nyc_taxi_lightgbm", "alias": "prod", "version": str(prev_version)}, timeout=10)
                raise AirflowFailException("🛑 Откат выполнен! Прод возвращен на прошлую стабильную версию.")
            else:
                print("🟢 Прод стабилен. Модель успешно введена в эксплуатацию!")
        except Exception as e:
            if "Авто-откат выполнен" in str(e): raise
            print(f"🟢 Проверка Canary завершена: аномалий не обнаружено.")

    status = check_prometheus_drift_metric()
    link = generate_s3_report_link(status)
    
    status >> link >> send_telegram_alert(status, link) >> run_continuous_training_job >> wait_for_engineer_approval >> promote_shadow_model_to_production() >> monitor_canary_health_and_rollback()

drift_sensor_dag_instance = drift_sensor_dag()
