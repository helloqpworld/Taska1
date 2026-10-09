FROM python:3.10-slim
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache-dir fastapi uvicorn httpx pydantic
COPY server.py /app/server.py
WORKDIR /app
EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
