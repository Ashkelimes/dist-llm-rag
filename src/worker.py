import os
import httpx
import logging
import time
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/worker.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=512, ge=1, le=4096)

class WorkerNode:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("MODEL_NAME", "phi3:mini")
        self.client = httpx.AsyncClient(timeout=300.0)  # Increased for queue wait times
        self.tasks_processed = 0
        logger.info(f"Worker {self.worker_id} initialized (Async mode) with model: {self.model_name}")

    async def process_task(self, request: InferenceRequest) -> dict:
        start_time = time.time()
        payload = {
            "model": self.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {"num_predict": request.max_tokens}
        }
        try:
            response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            latency = (time.time() - start_time) * 1000
            self.tasks_processed += 1
            return {
                "worker_id": self.worker_id,
                "generated_text": data.get("response", "").strip(),
                "latency_ms": round(latency, 2),
                "tokens_generated": data.get("eval_count", 0),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"[{self.worker_id}] Error: {str(e)}")
            return {"worker_id": self.worker_id, "status": "error", "message": str(e)}
