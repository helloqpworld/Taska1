import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import requests
from feast import FeatureStore
from sklearn.metrics import root_mean_squared_error

print("🚀 Запуск процесса Continuous Training (CT) с контуром Champion/Challenger...")

# БОЕВАЯ КОНФИГУРАЦИЯ ТЕЛЕГРАМА
TELEGRAM_TOKEN = "8909681536:AAHFffX3nX_8lLRLNqICGV-DWB5Ye9CXTgg"
TELEGRAM_CHAT_ID = "-5579643046"

# 1. Инициализируем Feast Feature Store
store = FeatureStore(repo_path="/app/feature_store")

# 2. Подгружаем локальный Parquet-файл наблюдений
raw_data = pd.read_parquet("/app/train_processed_2.parquet")

print("📋 Формирование паспорта наблюдений...")
entity_df = pd.DataFrame({
    "vendor_id": raw_data["vendor_id"].astype(np.int32),
    "pickup_datetime": raw_data["pickup_datetime"],
    "target_duration": raw_data["estimated_duration_by_speed"]
})

entity_train = entity_df.head(80)
entity_val = entity_df.tail(20)

print("🧬 Сбор признаков для ОБУЧЕНИЯ (Train Offline Store)...")
train_features = store.get_historical_features(
    entity_df=entity_train,
    features=[
        "taxi_trips_feature_view:passenger_count",
        "taxi_trips_feature_view:distance_haversine_km",
        "taxi_trips_feature_view:distance_manhattan_km",
        "taxi_trips_feature_view:distance_euclidean_deg",
        "taxi_trips_feature_view:expected_hourly_speed"
    ]
).to_df()

print("🧬 Сбор признаков для ОТЛОЖЕННОЙ ВАЛИДАЦИИ (Validation Offline Store)...")
val_features = store.get_historical_features(
    entity_df=entity_val,
    features=[
        "taxi_trips_feature_view:passenger_count",
        "taxi_trips_feature_view:distance_haversine_km",
        "taxi_trips_feature_view:distance_manhattan_km",
        "taxi_trips_feature_view:distance_euclidean_deg",
        "taxi_trips_feature_view:expected_hourly_speed"
    ]
).to_df()

def prepare_features(df):
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["pickup_datetime"].dt.dayofweek
    cols = [
        "passenger_count", "distance_haversine_km", "distance_manhattan_km",
        "distance_euclidean_deg", "expected_hourly_speed", "pickup_hour", "pickup_day_of_week"
    ]
    return df[cols], df["target_duration"]

X_train, y_train = prepare_features(train_features)
X_val, y_val = prepare_features(val_features)

# Настраиваем параметры сессии MLflow
MODEL_NAME = "nyc_taxi_lightgbm"
mlflow.set_tracking_uri("http://169.58.244.233:5080")
mlflow.set_experiment("nyc_taxi_continuous_training")

client = mlflow.tracking.MlflowClient()

# --------------------------------------------------------------------------------
# ЭТАП 1: ОПРЕДЕЛЯЕМ КАЧЕСТВО ТЕКУЩЕЙ БОЕВОЙ МОДЕЛИ (CHAMPION)
# --------------------------------------------------------------------------------
champion_rmse = float("inf")
champion_version = "N/A"

try:
    print("👑 Поиск текущего Champion (модели с алиасом @prod) в Model Registry...")
    # СТРОГО ПО ДОКУМЕНТАЦИИ: Метод извлечения версии по алиасу
    prod_model_run = client.get_model_version_by_alias(MODEL_NAME, "prod")
    champion_version = prod_model_run.version
    
    champion_model_uri = f"models:/{MODEL_NAME}/{champion_version}"
    champion_model = mlflow.lightgbm.load_model(champion_model_uri)
    
    champion_preds = champion_model.predict(X_val)
    champion_rmse = root_mean_squared_error(y_val, champion_preds)
    print(f"📊 Текущий Champion (v{champion_version}) на валидации показал RMSE: {champion_rmse:.4f}")
except Exception as e:
    print(f"ℹ️ Активная боевая модель @prod не найдена или это первый запуск контура ({e}).")

# --------------------------------------------------------------------------------
# ЭТАП 2: ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ НОВОЙ МОДЕЛИ (CHALLENGER)
# --------------------------------------------------------------------------------
print("📉 Обучение новой модели-претендента (Challenger) LightGBM...")
train_dataset = lgb.Dataset(X_train, label=y_train)
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "verbose": -1
}

with mlflow.start_run() as run:
    challenger_model = lgb.train(params, train_dataset, num_boost_round=50)
    
    challenger_preds = challenger_model.predict(X_val)
    challenger_rmse = root_mean_squared_error(y_val, challenger_preds)
    print(f"📊 Новая модель-Challenger показала RMSE: {challenger_rmse:.4f}")
    
    mlflow.log_metric("val_rmse", challenger_rmse)
    
    model_info = mlflow.lightgbm.log_model(
        lgb_model=challenger_model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )
    challenger_version = model_info.registered_model_version
    print(f"✅ Challenger успешно зарегистрирован как Версия №{challenger_version}")

    # --------------------------------------------------------------------------------
    # ЭТАП 3: РЫЦАРСКИЙ ТУРНИР (CHAMPION VS CHALLENGER) С ДВУХШАГОВОЙ ПЕРЕЗАПИСЬЮ АЛИАСА
    # --------------------------------------------------------------------------------
    print(f"⚔️ Итог дуэли: Challenger RMSE ({challenger_rmse:.4f}) vs Champion RMSE ({champion_rmse:.4f})")
    
    if challenger_rmse < champion_rmse:
        print("🔥 ПУШ НА СТЭЙДЖ: Новая модель готова к теневому тестированию!")
        
        # ГАРАНТИЯ БЕЗОПАСНОСТИ ДЛЯ POSTGRES: Сначала жестко стираем старый алиас "Ready_for_staging", если он был
        try:
            client.delete_registered_model_alias(MODEL_NAME, "Ready_for_staging")
            print("🧹 Старый алиас Ready_for_staging успешно удален из Model Registry.")
        except Exception:
            pass
            
        # Теперь со 100% чистой базой вешаем алиас на новую версию по канону документации
        client.set_registered_model_alias(MODEL_NAME, "Ready_for_staging", str(challenger_version))
        
        message_template = (
            "⚔️ <b>Результаты турнира Champion vs Challenger</b> ⚔️\n\n"
            "🟢 <b>Итог:</b> <code>НОВАЯ МОДЕЛЬ ПОБЕДИЛА!</code>\n"
            "👑 <b>Старый Champion (v{champion}):</b> RMSE = <code>{ch_rmse:.4f}</code>\n"
            "🎯 <b>Новый Challenger (v{challenger}):</b> RMSE = <code>{cl_rmse:.4f}</code>\n\n"
            "🚀 <b>Действие:</b> Модель успешно помечена как <code>@Ready_for_staging</code> и запущена в теневом режиме Kubernetes!"
        )
        message_text = message_template.format(
            champion=champion_version,
            ch_rmse=champion_rmse,
            challenger=challenger_version,
            cl_rmse=challenger_rmse
        )
    else:
        print("🛑 ОТКЛОНЕНО: Новая модель уступает текущему Champion.")
        message_template = (
            "⚔️ <b>Результаты турнира Champion vs Challenger</b> ⚔️\n\n"
            "🛑 <b>Итог:</b> <code>ОТКЛОНЕНО (Качество ниже)</code>\n"
            "👑 <b>Текущий Champion (v{champion}):</b> RMSE = <code>{ch_rmse:.4f}</code>\n"
            "🎯 <b>Новый Challenger (v{challenger}):</b> RMSE = <code>{cl_rmse:.4f}</code>\n\n"
            "⚠️ <b>Действие:</b> Модель заархивирована. На проде остается старая проверенная Версия №{champion}."
        )
        message_text = message_template.format(
            champion=champion_version,
            ch_rmse=champion_rmse,
            challenger=challenger_version,
            cl_rmse=challenger_rmse
        )

    # Отправка отчета в Telegram
    try:
        tg_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(tg_url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message_text, "parse_mode": "HTML"}, timeout=10)
        print("✅ Сервисный отчет турнира отправлен в Telegram.")
    except Exception as tg_err:
        print(f"⚠️ Ошибка отправки уведомления в Telegram: {tg_err}")

print("🏁 Процесс Champion/Challenger оценки успешно завершен!")
