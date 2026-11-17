FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_simulator.py .

RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home app \
 && chown -R app:app /app

USER app

CMD ["python", "data_simulator.py"]
