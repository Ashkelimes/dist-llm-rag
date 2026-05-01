import os
import asyncio
import logging
import time
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

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
    priority: int = Field(default=5, ge=1, le=10)  # 1=lowest, 10=highest

class TaskResponse(BaseModel):
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    result: Optional[dict] = None
    error: Optional[str] = None
    assigned_worker: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class MasterScheduler:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.workers: dict[str, dict] = {}  # worker_id -> metadata
        self.tasks: dict[str, TaskResponse] = {}
        self.running = False
        logger.info(f"Scheduler initialized with max {max_workers} workers")

    def register_worker(self, worker_id: str, metadata: dict):
        self.workers[worker_id] = {**metadata, "status": "idle", "last_heartbeat": time.time()}
        logger.info(f"Worker registered: {worker_id}")

    def assign_task(self, task_id: str, worker_id: str):
        if worker_id in self.workers:
            self.workers[worker_id]["status"] = "busy"
            self.tasks[task_id].assigned_worker = worker_id
            self.tasks[task_id].status = "running"
            logger.info(f"Task {task_id} assigned to {worker_id}")
            return True
        return False

    def complete_task(self, task_id: str, result: dict):
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].result = result
            self.tasks[task_id].completed_at = datetime.utcnow().isoformat()
            if self.tasks[task_id].assigned_worker:
                self.workers[self.tasks[task_id].assigned_worker]["status"] = "idle"
            logger.info(f"Task {task_id} completed")
            return True
        return False

    def fail_task(self, task_id: str, error: str):
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].error = error
            self.tasks[task_id].completed_at = datetime.utcnow().isoformat()
            if self.tasks[task_id].assigned_worker:
                self.workers[self.tasks[task_id].assigned_worker]["status"] = "idle"
            logger.warning(f"Task {task_id} failed: {error}")
            return True
        return False

    def get_status(self) -> dict:
        return {
            "queue_size": self.task_queue.qsize(),
            "active_workers": sum(1 for w in self.workers.values() if w["status"] == "busy"),
            "total_tasks": len(self.tasks),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.status == "completed"),
            "failed_tasks": sum(1 for t in self.tasks.values() if t.status == "failed")
        }

if __name__ == "__main__":
    # Quick local test
    import uuid
    sched = MasterScheduler()
    sched.register_worker("worker-1", {"model": "phi3:mini"})
    
    task_id = str(uuid.uuid4())[:8]
    req = TaskRequest(prompt="What is async I/O?", priority=8)
    resp = TaskResponse(
        task_id=task_id,
        status="pending",
        created_at=datetime.utcnow().isoformat()
    )
    sched.tasks[task_id] = resp
    sched.assign_task(task_id, "worker-1")
    
    print("Scheduler Status:", sched.get_status())
    print("Task State:", sched.tasks[task_id].model_dump())
