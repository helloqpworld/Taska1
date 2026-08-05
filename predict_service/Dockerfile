# 1. Забираем uv
FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# 2. Используем полный образ Python 3.11 (все системные библиотеки, включая libgomp1, уже внутри)
FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Копируем бинарник uv
COPY --from=uv_bin /uv /uvx /bin/

# Копируем только requirements
COPY requirements.txt .

# Установка: убрали ошибочный аргумент --prefer-binary.
# Оставляем только --system и --no-cache для минимизации размера.
RUN uv pip install --system --no-cache -r requirements.txt

# Копируем сам код приложения и HTML шаблоны
COPY . .

EXPOSE 8000

# Запускаем uvicorn на 1 воркере (надежный вариант с run_in_threadpool)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
