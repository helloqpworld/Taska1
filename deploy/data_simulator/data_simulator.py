"""
Симулятор потока поездок: генерация синтетических данных и запись в PostgreSQL.

Батчи случайного размера (в среднем 14 строк).
Генерируются со случайными паузами (в среднем 7 часов).
Записываются в таблицу trip_statistics через COPY.

Зависимости: numpy, pandas, psycopg2-binary, python-dotenv, prometheus-client
Переменные окружения (.env): DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
"""

from __future__ import annotations

import io
import logging
import os
import signal
import time
from contextlib import closing

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from prometheus_client import Gauge, start_http_server
from psycopg2 import sql

logger = logging.getLogger(__name__)

TABLE_NAME = "trip_statistics"

# --- Метрики Prometheus ---
# 1. Общее количество строк в таблице (обновляется после каждой вставки)
TOTAL_DB_ROWS = Gauge(
    "trip_generator_total_db_rows", "Всего строк в таблице trip_statistics"
)
# 2. Размер последнего сгенерированного батча (количество строк)
LAST_BATCH_SIZE = Gauge(
    "trip_generator_last_batch_size", "Строк в последнем сгенерированном батче"
)
# 3. Время ожидания до следующей генерации (в часах)
NEXT_SLEEP_HOURS = Gauge(
    "trip_generator_next_sleep_hours", "Часов до следующей генерации данных"
)

_INTERACTION_CATEGORIES = np.array(
    [
        "1_p1",
        "2_p1",
        "2_p2",
        "1_p2",
        "2_p5",
        "2_p6",
        "2_p3",
        "1_p3",
        "2_p4",
        "1_p4",
        "1_p5",
        "1_p6",
        "1_p0",
        "2_p0",
        "2_p7",
        "2_p8",
        "2_p9",
    ]
)
_INTERACTION_PROBS = np.array(
    [
        3.53375755e-01,
        3.09726275e-01,
        7.58197510e-02,
        5.91167707e-02,
        4.98792540e-02,
        3.09121262e-02,
        2.37090702e-02,
        1.47192046e-02,
        1.05604615e-02,
        7.18214037e-02,
        2.20704664e-04,
        9.75206654e-05,
        1.98890831e-05,
        1.86059164e-05,
        1.92474997e-06,
        6.41583325e-07,
        6.41583325e-07,
    ]
)
_INTERACTION_PROBS /= _INTERACTION_PROBS.sum()  # страховка от ошибок округления

_HOUR_PROBS = np.array(
    [
        0.036505137648391243,
        0.026443052588568560,
        0.019176714811838940,
        0.014324948376711522,
        0.010826493647524688,
        0.010284894737852416,
        0.022793772846561602,
        0.038117594149086415,
        0.045969407202854160,
        0.046387603829309960,
        0.044861528926866324,
        0.046944970808504335,
        0.049273846120095104,
        0.048999618824058510,
        0.050932235692876394,
        0.049231340889209430,
        0.044090950225003500,
        0.052434315706916830,
        0.062112482552288290,
        0.061912296626181580,
        0.057637093080971090,
        0.057714562292101430,
        0.055182758781443590,
        0.047842379634784084,
    ]
)
_HOUR_PROBS /= _HOUR_PROBS.sum()

_BEARING_WEIGHTS = np.array([40, 25, 23, 5, 5, 1, 0.5, 0.5])
_BEARING_WEIGHTS /= _BEARING_WEIGHTS.sum()


def _mixture(
    rng: np.random.Generator,
    size: int,
    weights: Sequence[float],
    samplers: Sequence[Callable[[int], np.ndarray]],
) -> np.ndarray:
    """Векторизованная выборка из смеси распределений.

    Компоненты выбираются сразу для всех строк, затем каждая компонента
    генерируется одним вызовом NumPy (вместо цикла с if/elif на каждую строку).
    """
    component = rng.choice(len(samplers), size=size, p=weights)
    values = np.empty(size)
    for i, sample in enumerate(samplers):
        mask = component == i
        if count := int(mask.sum()):
            values[mask] = sample(count)
    return values


def _direction_bearing(rng: np.random.Generator, size: int) -> np.ndarray:
    """Смесь распределений с отбраковкой значений вне [0, 353] (rejection sampling)."""
    samplers = (
        lambda n: rng.uniform(0.0, 353.0, n),
        lambda n: rng.normal(30.0, 25.0, n),
        lambda n: rng.normal(210.0, 25.0, n),
        lambda n: rng.normal(30.0, 0.4, n),
        lambda n: rng.normal(210.0, 0.4, n),
        lambda n: rng.normal(0.0, 0.2, n),
        lambda n: rng.normal(120.0, 0.6, n),
        lambda n: rng.normal(300.0, 0.6, n),
    )
    parts: list[np.ndarray] = []
    remaining = size
    while remaining:
        # Фильтр пропускает ~97% значений, поэтому берём небольшой запас.
        values = _mixture(rng, int(remaining * 1.1) + 8, _BEARING_WEIGHTS, samplers)
        values = values[(values >= 0.0) & (values <= 353.0)][:remaining]
        parts.append(values)
        remaining -= values.size
    return np.concatenate(parts)


def generate_dataframe(size: int, rng: np.random.Generator) -> pd.DataFrame:
    """Генерирует size строк синтетических данных о поездках (векторизованно)."""
    if size <= 0:
        raise ValueError(f"size должен быть положительным, получено {size}")

    speed_log = _mixture(
        rng,
        size,
        [0.75, 0.15, 0.03, 0.05, 0.02],
        (
            lambda n: rng.normal(1.65, 0.5, n),
            lambda n: rng.normal(2.1, 0.6, n),
            lambda n: rng.normal(3.15, 0.08, n),
            lambda n: rng.exponential(0.05, n),
            lambda n: rng.uniform(0.0, 4.0, n),
        ),
    ).clip(0.0, 3.9)

    speed = _mixture(
        rng,
        size,
        [0.11, 0.74, 0.15],
        (
            lambda n: rng.exponential(0.8, n),
            lambda n: rng.lognormal(1.7, 0.4, n),
            lambda n: rng.normal(21.5, 3.5, n),
        ),
    ).clip(0.0, 49.0)

    # Смеси нормальных распределений: параметры на строку выбираются по компоненте.
    minute_mean = rng.choice([5000.0, 6100.0], size=size, p=[0.4, 0.6])
    lon_component = rng.choice(4, size=size, p=[0.88, 0.08, 0.02, 0.02])

    return pd.DataFrame(
        {
            "vendor_id": rng.integers(1, 3, size),
            "passenger_count": np.rint(rng.triangular(0.0, 1.5, 9.0, size)).astype(int),
            "store_and_fwd_flag": rng.integers(0, 2, size),
            "trip_duration": rng.gamma(2.7, 200.0, size),
            "distance_haversine_km": rng.gamma(0.6, 6.0, size),
            "distance_manhattan_km": rng.gamma(0.9, 4.0, size),
            "distance_euclidean_deg": rng.gamma(1.3, 0.03, size) + 0.01,
            "centroid_latitude": rng.normal(40.7, 0.03, size),
            # Среднее двух независимых нормальных — как в исходном генераторе.
            "centroid_longitude": (
                rng.normal(-73.98, 0.01, size) + rng.normal(-74.0, 0.03, size)
            )
            / 2,
            "delta_latitude": rng.laplace(0.0, 0.02, size).clip(-0.5, 0.08),
            "time_of_day": rng.choice(
                [3, 1, 2, 0],
                size=size,
                p=[
                    0.3948358886746869,
                    0.2450748777631828,
                    0.24252799175124293,
                    0.11756124181088737,
                ],
            ),
            "estimated_duration_by_speed_log": speed_log,
            "minute_of_day": rng.normal(minute_mean, 100.0),
            "vendor_passenger_interaction": rng.choice(
                _INTERACTION_CATEGORIES, size=size, p=_INTERACTION_PROBS
            ),
            "is_weekend": rng.choice(
                [1, 0], size=size, p=[0.28535681084623804, 0.714643189153762]
            ),
            "estimated_duration_by_speed": speed,
            "direction_bearing": _direction_bearing(rng, size),
            "delta_longitude": rng.normal(
                np.array([0.0, 0.01, -0.12, -0.20])[lon_component],
                np.array([0.015, 0.035, 0.015, 0.015])[lon_component],
            ),
            "day_of_week": rng.integers(1, 7, size),  # как в исходнике: 1..6
            "expected_hourly_speed": rng.choice(
                [40, 15, 12, 25],
                size=size,
                p=[
                    0.047842379634784084,
                    0.09235701103216412,
                    0.11454679825920512,
                    0.7452538110738467,
                ],
            ),
            "hour": rng.choice(24, size=size, p=_HOUR_PROBS),
        }
    )


def _connect() -> psycopg2.extensions.connection:
    """Подключение к PostgreSQL по переменным окружения (fail fast при отсутствии пароля)."""
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise RuntimeError("DB_PASSWORD не задана")
    return psycopg2.connect(
        host=os.getenv(
            "DB_HOST", "db-0.db-service.postgres-namespace.svc.cluster.local"
        ),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "postgres"),
        password=password,
        dbname=os.getenv("DB_NAME", "postgres"),
        connect_timeout=10,
    )


def get_total_rows(conn: psycopg2.extensions.connection) -> int:
    """Возвращает точное количество строк через COUNT(*)."""
    with conn.cursor() as cursor:
        # Используем sql.Identifier для безопасного экранирования имени таблицы
        query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(TABLE_NAME))
        cursor.execute(query)
        return cursor.fetchone()[0]


def save_to_postgres(conn: psycopg2.extensions.connection, df: pd.DataFrame) -> None:
    """Пакетная вставка через COPY (текстовый CSV в памяти — самый быстрый путь в PostgreSQL).

    `with conn` — штатная транзакция psycopg2: commit при успехе, rollback при ошибке.
    Ошибки не глотаются, а поднимаются наверх — их обрабатывает вызывающий код.
    """
    if df.empty:
        logger.warning("Передан пустой DataFrame — сохранение пропущено")
        return

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)
    # sql.Identifier корректно экранирует имена таблицы и колонок.
    query = sql.SQL(
        r"COPY {} ({}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\t')"
    ).format(
        sql.Identifier(TABLE_NAME),
        sql.SQL(", ").join(map(sql.Identifier, df.columns)),
    )

    with conn:  # Автоматический commit/rollback
        with conn.cursor() as cursor:
            cursor.copy_expert(query.as_string(conn), buffer)
    logger.info("✅ В таблицу %s записано %d строк", TABLE_NAME, len(df))


def _sigterm(signum, frame):
    raise KeyboardInterrupt


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    rng = np.random.default_rng()

    # Запускаем HTTP-сервер для Prometheus на порту 9090
    start_http_server(9090)
    logger.info("Сервер метрик запущен на порту 9090")

    signal.signal(signal.SIGTERM, _sigterm)

    while True:
        conn = None
        try:
            # 1. Подключаемся
            logger.info("Подключение к БД ...")
            conn = _connect()
            logger.info("Подключение к БД успешно")

            # 2. Генерируем размер батча (с защитой от OOM)
            size = min(50000, max(1, int(rng.exponential(14))))
            LAST_BATCH_SIZE.set(size)  # <-- Обновляем метрику размера батча
            logger.info(f"Генерация {size} строк ...")

            df = generate_dataframe(size, rng)

            # 3. Сохраняем в БД
            logger.info("Сохранение в БД ...")
            save_to_postgres(conn, df)

            # 4. Считаем строки и обновляем метрику
            total = get_total_rows(conn)
            TOTAL_DB_ROWS.set(total)
            logger.info(f"Всего строк в БД: {total}")

        except psycopg2.OperationalError as e:
            logger.error(f"Ошибка соединения с БД: {e}")
        except KeyboardInterrupt:
            logger.info("Остановлено пользователем (SIGTERM/SIGINT)")
            break
        except Exception:
            logger.exception("Критический сбой в цикле")
            break
        finally:
            # 5. Гарантированно закрываем соединение
            if conn is not None:
                try:
                    conn.close()
                    logger.info("Соединение с БД закрыто")
                except Exception:
                    pass

        # 6. Вычисляем время сна и обновляем метрику
        sleep_in_hours = round(np.random.exponential(7), 0)
        sleep_in_sec = sleep_in_hours * 3600
        NEXT_SLEEP_HOURS.set(sleep_in_hours)  # <-- Обновляем метрику времени сна

        logger.info(f"Сон на {sleep_in_hours} часов\n")
        time.sleep(sleep_in_sec)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Фатальная ошибка")
        raise SystemExit(1)
