from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
)
from feast.value_type import ValueType
from feast.types import Float32, Int32

# ИСПРАВЛЕНО ДЛЯ FEAST 0.38+: Используем строго ValueType для Entity ключей!
vendor = Entity(
    name="vendor_id", 
    value_type=ValueType.INT32, 
    description="ID провайдера поездки такси"
)

# Источник сырых Parquet данных
taxi_source = FileSource(
    name="nyc_taxi_parquet_source",
    path="/home/alex/projects/ml-project/data/processed/train_processed_2.parquet",
    timestamp_field="pickup_datetime", # Главное поле для Point-in-Time контроля!
)

# Feature View со всеми базовыми физическими фичами модели
taxi_features_view = FeatureView(
    name="taxi_trips_feature_view",
    entities=[vendor],
    ttl=timedelta(days=365),
    schema=[
        Field(name="passenger_count", dtype=Int32),
        Field(name="store_and_fwd_flag", dtype=Int32),
        Field(name="distance_haversine_km", dtype=Float32),
        Field(name="distance_manhattan_km", dtype=Float32),
        Field(name="distance_euclidean_deg", dtype=Float32),
        Field(name="centroid_latitude", dtype=Float32),
        Field(name="centroid_longitude", dtype=Float32),
        Field(name="delta_latitude", dtype=Float32),
        Field(name="delta_longitude", dtype=Float32),
        Field(name="direction_bearing", dtype=Float32),
        Field(name="expected_hourly_speed", dtype=Int32),
        Field(name="estimated_duration_by_speed", dtype=Float32),
        Field(name="estimated_duration_by_speed_log", dtype=Float32),
    ],
    online=True,
    source=taxi_source,
)
