import sys

import numpy as np
import pandas as pd


def calculate_distances(df):
    # Переводим координаты в радианы для формулы Гаверсинуса
    lat1, lon1 = np.radians(df["pickup_latitude"]), np.radians(df["pickup_longitude"])
    lat2, lon2 = np.radians(df["dropoff_latitude"]), np.radians(df["dropoff_longitude"])

    # 1. Расстояние Гаверсинуса (в километрах)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["distance_haversine_km"] = 6371 * c

    # 2. Манхэттенское расстояние (в километрах)
    # Аппроксимация: 1 градус широты ~ 111 км, 1 градус долготы ~ 111 км * cos(latitude)
    lat_dist = np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) * 111.0
    lon_dist = (
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
        * 111.0
        * np.cos(np.radians(df["pickup_latitude"]))
    )
    df["distance_manhattan_km"] = lat_dist + lon_dist

    # 3. Евклидово расстояние (в градусах, полезно для деревьев)
    df["distance_euclidean_deg"] = np.sqrt(
        (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
        + (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
    )
    return df


def calculate_geometry(df):
    # 1. Центроид поездки (средняя точка)
    df["centroid_latitude"] = (df["pickup_latitude"] + df["dropoff_latitude"]) / 2
    df["centroid_longitude"] = (df["pickup_longitude"] + df["dropoff_longitude"]) / 2

    # 2. Простые разности координат (направление смещения)
    df["delta_latitude"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["delta_longitude"] = df["dropoff_longitude"] - df["pickup_longitude"]

    # 3. Направление движения (Bearing) в градусах от 0 до 360
    lat1, lon1 = np.radians(df["pickup_latitude"]), np.radians(df["pickup_longitude"])
    lat2, lon2 = np.radians(df["dropoff_latitude"]), np.radians(df["dropoff_longitude"])

    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)

    bearing = np.degrees(np.arctan2(y, x))
    df["direction_bearing"] = (bearing + 360) % 360

    return df


def generate_advanced_features(df):
    # 1. Разбор даты и времени
    df["hour"] = df["pickup_datetime"].dt.hour.astype("uint8")
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek.astype("uint8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("uint8")
    df["minute_of_day"] = df["hour"] * 60 + df["pickup_datetime"].dt.minute
    df["minute_of_day"] = df["minute_of_day"].astype("uint8")

    # 2. Категории времени суток (полезно для линейной модели)
    # 0 - ночь, 1 - утро, 2 - день, 3 - вечер
    df["time_of_day"] = pd.cut(
        df["hour"], bins=[-1, 5, 11, 16, 23], labels=[0, 1, 2, 3]
    ).astype("uint8")

    # 5. СИНЕРГИЯ 2: Ожидаемая скорость (Прокси-признак)
    # В часы пик скорость ниже. Создаем коэффициент «загруженности» часа.
    # Для линейной модели это даст нелинейную подсказку.
    # (Вы вычисляете среднюю скорость по часам на ТРЕЙНЕ, тут пример маппинга)
    rush_hour_map = {
        8: 15,
        9: 15,
        17: 12,
        18: 12,
        23: 40,
        12: 25,
    }  # примерная скорость в км/ч
    df["expected_hourly_speed"] = (
        df["hour"].map(rush_hour_map).fillna(25).astype("uint8")
    )

    # Расчет приблизительного времени на основе исторической скорости часа пик
    df["estimated_duration_by_speed"] = (
        df["distance_haversine_km"] / df["expected_hourly_speed"]
    ) * 60  # в минутах

    # ДОБАВИТЬ ЭТУ СТРОКУ: логарифмируем расчетное время для синергии с логом таргета
    df["estimated_duration_by_speed_log"] = np.log1p(df["estimated_duration_by_speed"])

    # 6. СИНЕРГИЯ 3: Вместимость вендора
    df["vendor_passenger_interaction"] = (
        df["vendor_id"].astype(str) + "_p" + df["passenger_count"].astype(str)
    )

    return df
