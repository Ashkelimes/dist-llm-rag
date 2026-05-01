import os
import httpx
import logging
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Configure logging to both console and file
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
        self.tasks_processed = 0
        self.total_latency = 0.0
        logger.info(f"Worker {self.worker_id} initialized with model: {self.model_name}")

    def process_task(self, request: InferenceRequest) -> dict:
        logger.info(f"[{self.worker_id}] Processing: {request.prompt[:40]}...")
        start_time = time.time()
        payload = {
            "model": self.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {"num_predict": request.max_tokens}
        }
        try:
            response = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()
            latency = time.time() - start_time
            self.tasks_processed += 1
            self.total_latency += latency
            return {
                "worker_id": self.worker_id,
                "generated_text": data.get("response", "").strip(),
                "latency_ms": round(latency * 1000, 2),
                "tokens_generated": data.get("eval_count", 0),
                "status": "success"
            }
        except httpx.TimeoutException:
            logger.error(f"[{self.worker_id}] Ollama timeout")
            return {"worker_id": self.worker_id, "status": "error", "message": "timeout"}
        except Exception as e:
            logger.error(f"[{self.worker_id}] Error: {str(e)}")
            return {"worker_id": self.worker_id, "status": "error", "message": str(e)}

if __name__ == "__main__":
    # Quick local test
    w = WorkerNode("test-1")
    req = InferenceRequest(prompt="Explain caching in 5 words.", max_tokens=20)
    print(w.process_task(req))
