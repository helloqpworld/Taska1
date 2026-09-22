import io
import time
import random
import requests
import pandas as pd
import numpy as np

API_URL = "http://127.0.0.1:80/predict"
HEADERS = {"Host": "predict.local"}

TOTAL_STEPS = 15
SLEEP_INTERVAL = 4.0

BASE_ROW = {
    "vendor_id": 2,
    "passenger_count": 1,
    "store_and_fwd_flag": 0,
    "distance_haversine_km": 1.4985207,
    "distance_manhattan_km": 1.732391,
    "distance_euclidean_deg": 0.0176795,
    "centroid_latitude": 40.766769,
    "centroid_longitude": -73.973388,
    "delta_latitude": -0.0023345,
    "delta_longitude": 0.0175247,
    "direction_bearing": 99.97019,
    "hour": 17,
    "day_of_week": 0,
    "is_weekend": 0,
    "minute_of_day": 20,
    "time_of_day": 3,
    "expected_hourly_speed": 12,
    "estimated_duration_by_speed": 7.492603,
    "estimated_duration_by_speed_log": 2.139195
}

def generate_batch(drift_factor: float, batch_size: int = 3) -> bytes:
    rows = []
    drifted_lat = BASE_ROW["centroid_latitude"] + (34.0522 - 40.7667) * drift_factor
    drifted_lon = BASE_ROW["centroid_longitude"] + (-118.2437 - (-73.9733)) * drift_factor
    drifted_speed = int(BASE_ROW["expected_hourly_speed"] + (drift_factor * 60.0))

    for _ in range(batch_size):
        row = BASE_ROW.copy()
        row["hour"] = random.randint(0, 23)
        row["minute_of_day"] = random.randint(0, 255)

        if drift_factor > 0.1:
            row["centroid_latitude"] = drifted_lat
            row["centroid_longitude"] = drifted_lon
            row["expected_hourly_speed"] = drifted_speed
            row["distance_haversine_km"] *= (1.0 + drift_factor * 5.0)

        rows.append(row)

    df = pd.DataFrame(rows)

    # ИСПРАВЛЕНО: Приводим строго к int32 и float32, как требует сигнатура MLflow
    for col in df.columns:
        if df[col].dtype == np.float64 or "distance" in col or "latitude" in col or "longitude" in col or "bearing" in col or "delta" in col or "duration" in col:
            df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype(np.int32) # Переводим int64 строго в нужный int32!

    # Сохраняем в бинарный Parquet буфер, удерживающий типы данных
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine="pyarrow")
    return parquet_buffer.getvalue()

print("🚀 Запуск Parquet-генератора нагрузки через парадный Ingress (Порт 80)...")

try:
    for step in range(TOTAL_STEPS):
        drift_factor = step / TOTAL_STEPS
        #drift_factor = 0.0
        parquet_data = generate_batch(drift_factor=drift_factor, batch_size=3)

        # Меняем имя файла на .parquet, чтобы FastAPI правильно выбрал read_parquet
        files = {'file': ('nyc_taxi_drift.parquet', io.BytesIO(parquet_data), 'application/octet-stream')}

        start_time = time.time()
        try:
            response = requests.post(API_URL, headers=HEADERS, files=files, timeout=30, allow_redirects=False)
            # print(f"[💥 Шаг {step+1}] Ответ сервера: {response.text}")
            print(f"[Шаг {step+1}/{TOTAL_STEPS}] Дрифт фичей: {drift_factor:.1%} | "
                  f"Статус API: {response.status_code} | Время обработки: {time.time() - start_time:.2f} sec")

        except requests.exceptions.RequestException as e:
            print(f"[Шаг {step+1}/{TOTAL_STEPS}] Ошибка сети: {e}")

        time.sleep(SLEEP_INTERVAL)

except KeyboardInterrupt:
    print("\n🛑 Симуляция остановлена.")

print("🏁 Нагрузочное тестирование успешно завершено!")
