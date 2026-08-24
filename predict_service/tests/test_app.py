import pytest

# Корректно импортируйте ваш app. Напрямую зависит от структуры внутри predict_service/app
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_main():
    response = client.get(
        "/"
    )  # замените на ваш реальный эндпоинт, например /health или /docs
    assert response.status_code in [200, 404]
