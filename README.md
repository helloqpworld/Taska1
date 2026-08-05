# Classic ML & MLOps Infrastructure Project
!!! README НУЖДАЕТСЯ В ДОРАБОТКЕ !!!

Проект сквозного MLOps-контура: от обучения моделей до production-ready сервиса предсказаний.

## 🏗 Стек технологий
* **ML Pipeline**: `Python 3.11`, `LightGBM`, `XGBoost`, `Scikit-Learn`.
* **Окружение и зависимости**: `uv` (Rust-менеджер пакетов).
* **Версионирование данных**: `DVC` (Data Version Control).
* **Инфраструктура**: `MLflow Server` + `PostgreSQL` (трекинг), `MinIO Enterprise` (S3-хранилище).
* **Inference API**: `FastAPI` (микросервис) + `Jinja2` (UI-страница).
* **Качество кода**: `pre-commit` хуки (`black`, `isort`).

---

## 📁 Структура проекта
```text
.
├── data/                           # Данные (отслеживаются через DVC)
│   ├── raw/                        # Сырые датасеты
│   └── processed/                  # Подготовленные признаки
├── src/                            # Код ML-пайплайна
│   ├── data_prep.py                # Очистка данных
│   ├── features.py                 # Генерация фичей
│   ├── train.py                    # Обучение и логирование в MLflow
│   └── predict.py                  # Локальный пакетный инференс
├── predict_service/                # [Контур предсказаний] Сервис FastAPI
│   ├── app/main.py                 # Логика сервера и загрузка модели в RAM
│   ├── templates/index.html        # HTML-страница с кнопкой загрузки
│   ├── requirements.txt            # Зависимости инференса (включая pyarrow)
│   ├── Dockerfile                  # Оптимизированный Docker-образ на uv
│   └── docker-compose.yml          # Запуск контейнера FastAPI
├── docker-compose.yml              # [Контур обучения] Стек (Postgres, MinIO, MLflow)
├── pyproject.toml / uv.lock        # Конфигурация проекта и зависимостей хоста
└── .pre-commit-config.yaml         # Настройки линтеров перед коммитом
```

---

## 🚀 Быстрый старт

### 1. Подготовка секретов
Создайте файлы `.env` в корне проекта и в папке `predict_service/` на основе шаблонов `.env.example`, заполнив пароли к базам и S3.

### 2. Запуск инфраструктуры (Контур обучения)
В корне проекта поднимите базы и MLflow:
```bash
docker compose up -d
```
*Интерфейсы: MLflow UI — `http://localhost:5050`, MinIO Console — `http://localhost:9001`.*

### 3. Обучение и регистрация модели
1. Стяните данные и разверните окружение хоста:
   ```bash
   dvc pull
   uv venv --python 3.11 && source .venv/bin/activate && uv sync
   pre-commit install
   ```
2. Запустите обучение (модель автоматически улетит в MinIO):
   ```bash
   python src/train.py
   ```
3. В **MLflow UI** зарегистрируйте модель под именем `my_best_model` и присвойте целевой версии алиас **`prod`**.

### 4. Запуск сервиса предсказаний (Контур инференса)

#### Вариант А: Локальный запуск (На той же машине)
В `predict_service/docker-compose.yml` используется `network_mode: "host"`. FastAPI видит базы через `localhost`.
```bash
cd predict_service
docker compose up --build -d
```
*Интерфейс: `http://localhost:8000`*

#### Вариант Б: Продакшн запуск (На удаленном сервере)
В `.env` укажите реальный IP сервера баз (`MAIN_SERVER_IP=10.70.115.153`). В `docker-compose.yml` раскомментируйте проброс портов `8000:8000` и закомментируйте `network_mode`.
```bash
docker compose up --build -d
```
*Интерфейс: `http://<IP_СЕРВЕРА_ФАСТАПИ>:8000`*

---

## 💻 Работа с интерфейсом
1. Откройте страницу приложения в браузере.
2. Выберите файл формата **`.csv`** или **`.parquet`** (колонки должны совпадать с фичами модели).
3. Нажмите **«Загрузить и предсказать»**.
4. Браузер автоматически скачает готовый файл с добавленной колонкой **`prediction`**.
