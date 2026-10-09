from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import httpx
import os

app = FastAPI()

# Используем проверенный IP и порт туннеля!
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://172.17.0.1:8081")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

class SeldonRequest(BaseModel):
    data: Optional[Dict[str, Any]] = None
    inputs: Optional[List[Dict]] = None

@app.get("/ready")
async def ready():
    return {"status": "ready"}

@app.post("/v2/models/vllm-proxy/infer")
async def predict(request: SeldonRequest):
    try:
        # Извлекаем промпт из формата Seldon V2
        prompt = "Привет!"
        if request.data and "ndarray" in request.data:
            prompt = request.data["ndarray"][0][0]
        elif request.inputs:
            prompt = request.inputs[0].get("data", ["Привет!"])[0]

        # Формируем запрос ровно так, как мы успешно тестировали
        vllm_payload = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{VLLM_ENDPOINT}/v1/chat/completions",
                json=vllm_payload
            )
            response.raise_for_status()
            result = response.json()

        generated_text = result["choices"][0]["message"]["content"]

        # Возвращаем ответ в формате Seldon V2
        return {
            "model_name": "vllm-proxy",
            "model_version": "v1",
            "outputs": [
                {
                    "name": "output",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [generated_text],
                }
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM Error: {str(e)}")
