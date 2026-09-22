import io
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# Импортируем инструменты Prometheus
import prometheus_client
from prometheus_fastapi_instrumentator import Instrumentator

# Глобальный словарь для хранения модели в оперативной памяти (RAM)
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Старт приложения: Загрузка модели из MLflow/MinIO...")
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        model_name = "my_best_model"
        model_uri = f"models:/{model_name}@prod"
        mlflow.set_tracking_uri(mlflow_uri)

        ml_models["predict_model"] = await run_in_threadpool(
            mlflow.pyfunc.load_model, model_uri
        )
        print("Модель успешно загружена в память и готова к работе!")
    except Exception as e:
        print(f"ВНИМАНИЕ: Модель '{model_name}' с алиасом '@prod' не найдена в MLflow. Детали: {e}")
        ml_models["predict_model"] = None
    yield
    print("Остановка приложения: Очистка ресурсов...")
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# ================================================================================
#  ЯВНЫЙ СБОР И ЭКСПОРТ МЕТРИК ДЛЯ KUBERNETES (ОБХОД БЛОКИРОВКИ LIFESPAN)
# ================================================================================
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*", "/metrics"],
    env_var_name="ENABLE_METRICS",
)
instrumentator.instrument(app)

ML_DRIFT_COUNT = prometheus_client.Gauge('ml_drifted_features_count', 'Количество фичей с обнаруженным дрифтом')
ML_DRIFT_SHARE = prometheus_client.Gauge('ml_drifted_features_share', 'Доля фичей с дрифтом от общего числа')
ML_DATASET_DRIFT = prometheus_client.Gauge('ml_dataset_drift_alert', 'Флаг критического сдвига всего датасета (0 или 1)')

@app.get("/metrics", tags=["monitoring"])
async def metrics_endpoint():
    """Генерирует и отдает метрики в нативном текстовом формате Prometheus."""
    return Response(
        content=prometheus_client.generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

@app.post("/internal/drift-report", tags=["monitoring"])
async def receive_drift_report(data: dict):
    """Принимает метрики дрифта от скрипта детектора и обновляет их в Prometheus."""
    ML_DRIFT_COUNT.set(data.get("number_of_drifted_features", 0))
    ML_DRIFT_SHARE.set(data.get("share_of_drifted_features", 0.0))
    ML_DATASET_DRIFT.set(1 if data.get("dataset_drift", False) else 0)
    return {"status": "success", "message": "ML metrics updated successfully"}
# ================================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".parquet")):
        raise HTTPException(status_code=400, detail="Допускаются только файлы .csv и .parquet")

    contents = await file.read()
    loaded_model = ml_models.get("predict_model")
    if not loaded_model:
        raise HTTPException(
            status_code=503,
            detail="Сервис временно недоступен: ML-модель еще не обучена или не зарегистрирована.",
        )

    try:
        if filename_lower.endswith(".csv"):
            df = await run_in_threadpool(pd.read_csv, io.BytesIO(contents))
        else:
            df = await run_in_threadpool(pd.read_parquet, io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать файл: {e}")

    try:
        predictions = await run_in_threadpool(loaded_model.predict, df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ошибка при предсказании модели: {e}")

    df["prediction"] = predictions

    if filename_lower.endswith(".csv"):
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        media_type = "text/csv"
        return_content = stream.getvalue()
    else:
        stream = io.BytesIO()
        df.to_parquet(stream, index=False, engine="pyarrow")
        media_type = "application/octet-stream"
        return_content = stream.getvalue()

    response = StreamingResponse(iter([return_content]), media_type=media_type)
    response.headers["Content-Disposition"] = f"attachment; filename=predictions_{file.filename}"
    return response
