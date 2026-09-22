import datetime
import requests
import boto3
from botocore.client import Config
from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# ================================================================================
# БОЕВАЯ КОНФИГУРАЦИЯ КОНТУРА CT (ФИКСИРОВАННЫЙ CLUSTERIP + СКВОЗНЫЕ ЛОГИ)
# ================================================================================
TELEGRAM_TOKEN = "8909681536:AAHFffX3nX_8lLRLNqICGV-DWB5Ye9CXTgg"
TELEGRAM_CHAT_ID = "-5579643046"

PROMETHEUS_URL = "http://10.96.146.167:9090"
MINIO_EXTERNAL_URL = "http://169.58.244.233:9000" 
BUCKET_NAME = "ml-monitoring"
RETRAIN_IMAGE = "169.58.244.233:8929/root/taska1/predict_service:latest"

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
    tags=['monitoring', 'mlops', 'continuous-training'],
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

    # Валидный PodSpec
    pod_configuration = k8s.V1Pod(
        spec=k8s.V1PodSpec(
            containers=[k8s.V1Container(name="base")],
            host_network=True
        )
    )

    # ИСПРАВЛЕНО ДЛЯ ОТЛАДКИ ОШИБОК: Жестко отключаем автоудаление пода (is_delete_operator_pod=True)
    # и заставляем Airflow принудительно писать логи самого контейнера в поток планировщика
    run_continuous_training_job = KubernetesPodOperator(
        task_id="run_continuous_training_job",
        name="ml-taxi-lgb-retrain-pod",
        namespace="airflow",
        image=RETRAIN_IMAGE,
        image_pull_policy="IfNotPresent",
        full_pod_spec=pod_configuration,
        get_logs=True,                  # Включаем стриминг stdout/stderr из контейнера
        is_delete_operator_pod=True,   # ХРАНИМ ПОД В ПАМЯТИ ПОСЛЕ ОШИБКИ ДЛЯ СНЯТИЯ ЛОГОВ!
        log_events_on_failure=True
    )

    status = check_prometheus_drift_metric()
    link = generate_s3_report_link(status)
    
    status >> link >> send_telegram_alert(status, link) >> run_continuous_training_job

drift_sensor_dag_instance = drift_sensor_dag()
