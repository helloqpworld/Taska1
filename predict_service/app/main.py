import io
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# Глобальный словарь для хранения модели в оперативной памяти (RAM)
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Старт приложения: Загрузка модели из MLflow/MinIO...")
    try:
        # Так как контейнер запущен в режиме network_mode: host,
        # мы гарантированно стучимся локально на порт хоста
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        model_name = "my_best_model"
        model_uri = f"models:/{model_name}@prod"

        mlflow.set_tracking_uri(mlflow_uri)

        # Скачиваем модель в RAM строго ОДИН РАЗ при старте всего контейнера
        # Выносим синхронную загрузку MLflow в пул потоков, чтобы не вешать Event Loop
        ml_models["predict_model"] = await run_in_threadpool(
            mlflow.pyfunc.load_model, model_uri
        )
        print("Модель успешно загружена в память и готова к работе!")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА при загрузке модели: {e}")
        raise e

    yield

    print("Остановка приложения: Очистка ресурсов...")
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

# Надежный абсолютный путь к шаблонам для Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    # Универсальный синтаксис, подходящий для любых новых версий FastAPI/Starlette
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".parquet")):
        raise HTTPException(
            status_code=400, detail="Допускаются только файлы .csv и .parquet"
        )

    contents = await file.read()

    loaded_model = ml_models.get("predict_model")
    if not loaded_model:
        raise HTTPException(
            status_code=500, detail="Модель не инициализирована в памяти."
        )

    # ВАЖНО: Чтение тяжелых файлов и синхронный метод .predict()
    # мы выполняем в пуле потоков через run_in_threadpool,
    # чтобы сервер не зависал при одновременных запросах.
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
        raise HTTPException(
            status_code=422, detail=f"Ошибка при предсказании модели: {e}"
        )

    df["prediction"] = predictions

    # Формируем ответ обратно в исходном формате
    if filename_lower.endswith(".csv"):
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        media_type = "text/csv"
        return_content = stream.getvalue()
    else:
        stream = io.BytesIO()
        # Явно фиксируем движок pyarrow, который необходим для стабильной работы
        df.to_parquet(stream, index=False, engine="pyarrow")
        media_type = "application/octet-stream"
        return_content = stream.getvalue()

    response = StreamingResponse(iter([return_content]), media_type=media_type)
    response.headers["Content-Disposition"] = (
        f"attachment; filename=predictions_{file.filename}"
    )
    return response
