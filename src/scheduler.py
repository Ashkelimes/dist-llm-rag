import os, sys, asyncio, logging, time, random
from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import WorkerNode, InferenceRequest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('logs/scheduler.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TaskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    priority: int = Field(default=5, ge=1, le=10)
    use_rag: bool = Field(default=False)

class MasterScheduler:
    def __init__(self, num_workers: int = 4, model_name: str = "phi3:mini", max_retries: int = 2, fault_injection_rate: float = 0.0):
        self.num_workers = num_workers
        self.max_retries = max_retries
        self.fault_injection_rate = fault_injection_rate
        self.semaphore = asyncio.Semaphore(4)
        self.workers: List[WorkerNode] = []
        self.next_worker_idx = 0
        self.metrics = {"total_assigned": 0, "total_completed": 0, "total_failed": 0, "reassigned": 0, "avg_latency_ms": 0.0}
        self.worker_health = {f"worker-{i+1}": {"status": "healthy", "failures": 0} for i in range(num_workers)}
        self._init_workers(model_name)
        logger.info(f"Scheduler: {num_workers} workers | Retries={max_retries} | FaultInjection={fault_injection_rate*100:.0f}%")

    def _init_workers(self, model_name: str):
        for i in range(self.num_workers): self.workers.append(WorkerNode(f"worker-{i+1}"))

    def _get_next_worker(self) -> WorkerNode:
        worker = self.workers[self.next_worker_idx]
        self.next_worker_idx = (self.next_worker_idx + 1) % len(self.workers)
        return worker

    def _record_failure(self, worker_id: str):
        self.worker_health[worker_id]["failures"] += 1
        if self.worker_health[worker_id]["failures"] >= 3:
            self.worker_health[worker_id]["status"] = "degraded"
            logger.warning(f"WORKER_DEGRADED: {worker_id} marked unhealthy")

    def _record_success(self, worker_id: str):
        self.worker_health[worker_id]["failures"] = 0
        self.worker_health[worker_id]["status"] = "healthy"

    async def process_request(self, request: TaskRequest) -> dict:
        self.metrics["total_assigned"] += 1
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            worker = self._get_next_worker()
            async with self.semaphore:
                logger.info(f"ATTEMPT {attempt+1}/{self.max_retries} -> {worker.worker_id}")
                try:
                    # FAULT INJECTION (for testing only)
                    if random.random() < self.fault_injection_rate:
                        raise ConnectionError(f"Simulated node crash on {worker.worker_id}")
                    
                    task = InferenceRequest(prompt=request.prompt, max_tokens=request.max_tokens)
                    result = await worker.process_task(task)
                    
                    if result.get("status") == "success":
                        self.metrics["total_completed"] += 1
                        self._record_success(worker.worker_id)
                        self._update_latency(result.get("latency_ms", 0))
                        result["used_rag"] = request.use_rag
                        if attempt > 0:
                            self.metrics["reassigned"] += 1
                            logger.info(f"? RECOVERY: Task reassigned from failed attempt to {worker.worker_id} | SUCCESS")
                        return result
                except Exception as e:
                    last_error = str(e)
                    self._record_failure(worker.worker_id)
                    logger.warning(f"? TASK_FAILED: {worker.worker_id} -> {last_error}")
            attempt += 1

        self.metrics["total_failed"] += 1
        logger.critical(f"?? PERMANENT_FAILURE: Max retries exhausted. Error: {last_error}")
        return {"status": "error", "message": last_error or "Worker failure", "worker_id": "unknown"}

    def _update_latency(self, latency_ms: float):
        total = self.metrics["total_completed"]
        if total > 0:
            prev = self.metrics["avg_latency_ms"]
            self.metrics["avg_latency_ms"] = prev + ((latency_ms - prev) / total)

    def get_status(self) -> dict:
        return {**self.metrics, "worker_health": {k: v["status"] for k,v in self.worker_health.items()}, "timestamp": datetime.now(timezone.utc).isoformat()}
