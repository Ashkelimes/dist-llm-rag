import os
import sys
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import WorkerNode, InferenceRequest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    priority: int = Field(default=5, ge=1, le=10)
    use_rag: bool = Field(default=False)

class MasterScheduler:
    def __init__(self, num_workers: int = 4, model_name: str = "phi3:mini", max_concurrent_ollama: int = 3):
        self.num_workers = num_workers
        self.max_concurrent_ollama = max_concurrent_ollama  # ADAPTIVE: limit Ollama calls
        self.semaphore = asyncio.Semaphore(max_concurrent_ollama)
        self.workers: List[WorkerNode] = []
        self.next_worker_idx = 0
        self.metrics = {"total_assigned": 0, "total_completed": 0, "total_failed": 0, "avg_latency_ms": 0.0, "queued": 0}
        self._init_workers(model_name)
        logger.info(f"Scheduler: Adaptive backpressure (max {max_concurrent_ollama} concurrent Ollama calls)")

    def _init_workers(self, model_name: str):
        for i in range(self.num_workers):
            self.workers.append(WorkerNode(f"worker-{i+1}"))

    def _get_next_worker(self) -> WorkerNode:
        worker = self.workers[self.next_worker_idx]
        self.next_worker_idx = (self.next_worker_idx + 1) % len(self.workers)
        return worker

    async def process_request(self, request: TaskRequest) -> dict:
        self.metrics["total_assigned"] += 1
        
        # ADAPTIVE BACKPRESSURE: Wait for slot, don't timeout
        self.metrics["queued"] += 1
        logger.debug(f"Request #{self.metrics['total_assigned']} queued (waiting for Ollama slot)...")
        
        async with self.semaphore:
            worker = self._get_next_worker()
            logger.info(f"Request #{self.metrics['total_assigned']} executing on {worker.worker_id}")
            
            try:
                # Tiny delay to let GPU recover between batches (addresses 425+ failure pattern)
                await asyncio.sleep(0.05)
                
                task = InferenceRequest(prompt=request.prompt, max_tokens=request.max_tokens)
                result = await worker.process_task(task)
                
                if result.get("status") == "success":
                    self.metrics["total_completed"] += 1
                    self._update_latency(result.get("latency_ms", 0))
                    result["used_rag"] = request.use_rag
                    return result
            except Exception as e:
                self.metrics["total_failed"] += 1
                logger.error(f"Request #{self.metrics['total_assigned']} failed on {worker.worker_id}: {e}")
                return {"status": "error", "message": str(e), "worker_id": worker.worker_id}
        
        # Fallback if semaphore wait fails (shouldn't happen)
        self.metrics["total_failed"] += 1
        return {"status": "error", "message": "Semaphore timeout", "worker_id": "unknown"}

    def _update_latency(self, latency_ms: float):
        total = self.metrics["total_completed"]
        if total > 0:
            prev_avg = self.metrics["avg_latency_ms"]
            self.metrics["avg_latency_ms"] = prev_avg + ((latency_ms - prev_avg) / total)

    def get_status(self) -> dict:
        return {
            **self.metrics,
            "active_workers": self.num_workers,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

if __name__ == "__main__":
    import asyncio
    sched = MasterScheduler(max_concurrent_ollama=2)
    async def test():
        req = TaskRequest(prompt="What is adaptive backpressure?", max_tokens=20)
        print(await sched.process_request(req))
    asyncio.run(test())
