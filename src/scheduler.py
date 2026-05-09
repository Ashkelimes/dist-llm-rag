import os, sys, asyncio, logging, time, random
from datetime import datetime, timezone
from typing import List, Dict
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logging_config import setup_component_logger
from worker import WorkerNode, InferenceRequest
from rag_module import RAGPipeline

logger = setup_component_logger("scheduler")

class TaskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    priority: int = Field(default=5, ge=1, le=10)
    use_rag: bool = Field(default=False)

class MasterScheduler:
    def __init__(self, num_workers: int = 4, model_name: str = "phi3:mini", max_retries: int = 2,
                 fault_rate: float = 0.0, use_rag: bool = False, routing_strategy: str = "least_connections"):
        self.num_workers = num_workers
        self.max_retries = max_retries
        self.fault_rate = fault_rate
        self.use_rag = use_rag
        self.routing_strategy = routing_strategy
        self.semaphore = asyncio.Semaphore(4)
        self.workers: List[WorkerNode] = []
        self.next_worker_idx = 0
        self.active_requests: Dict[str, int] = {f"worker-{i+1}": 0 for i in range(num_workers)}
        self.metrics = {"assigned": 0, "completed": 0, "failed": 0, "reassigned": 0, "avg_lat": 0.0}
        self.worker_health = {f"worker-{i+1}": {"status": "healthy", "failures": 0, "hb_time": time.time()} for i in range(num_workers)}
        self._state_lock = asyncio.Lock()
        self._hb_task = None
        self._hb_started = False
        if use_rag:
            self.rag = RAGPipeline()
            logger.info("[RAG] Pipeline initialized")
        self._init_workers(model_name)
        logger.info(f"[INIT] Scheduler ready | Workers: {num_workers} | Strategy: {routing_strategy} | Retries: {max_retries}")

    def _init_workers(self, model_name: str):
        for i in range(self.num_workers): self.workers.append(WorkerNode(f"worker-{i+1}"))

    async def start_health_monitor(self):
        if not self._hb_started:
            self._hb_task = asyncio.create_task(self._hb_loop())
            self._hb_started = True
            logger.info("[HEARTBEAT] Monitor started (polls every 30s)")

    async def _hb_loop(self):
        logger.info("[HEARTBEAT] Background loop active")
        while True:
            try:
                for w in self.workers:
                    prev_status = self.worker_health[w.worker_id]["status"]
                    ok = await w.health_check()
                    
                    # FIX #2: Protect ALL worker_health mutations with _state_lock
                    async with self._state_lock:
                        self.worker_health[w.worker_id]["hb_time"] = time.time()
                        if ok:
                            self.worker_health[w.worker_id]["failures"] = 0
                            self.worker_health[w.worker_id]["status"] = "healthy"
                            if prev_status == "degraded":
                                logger.info(f"[HEARTBEAT] RECOVERY | {w.worker_id} back to healthy")
                        else:
                            self.worker_health[w.worker_id]["failures"] += 1
                            logger.warning(f"[HEARTBEAT] FAIL | {w.worker_id} | Consecutive: {self.worker_health[w.worker_id]['failures']}")
                            if self.worker_health[w.worker_id]["failures"] >= 2:
                                self.worker_health[w.worker_id]["status"] = "degraded"
                                logger.warning(f"[HEARTBEAT] FAULT | {w.worker_id} marked DEGRADED")
                await asyncio.sleep(30)
            except asyncio.CancelledError: break
            except Exception as e: logger.error(f"[HEARTBEAT] Loop error: {e}"); await asyncio.sleep(10)

    def _get_next_worker(self) -> WorkerNode:
        if not self._hb_started: asyncio.create_task(self.start_health_monitor())
        
        if self.routing_strategy == "least_connections":
            healthy = [w for w in self.workers if self.worker_health[w.worker_id]["status"] != "degraded"]
            if not healthy: healthy = self.workers
            selected = min(healthy, key=lambda w: self.active_requests.get(w.worker_id, 999))
            logger.info(f"[ROUTING] Strategy={self.routing_strategy} | Selected={selected.worker_id} | ActiveReqs={self.active_requests.get(selected.worker_id, 0)}")
            return selected
        else:
            w = self.workers[self.next_worker_idx]
            self.next_worker_idx = (self.next_worker_idx + 1) % len(self.workers)
            logger.info(f"[ROUTING] Strategy={self.routing_strategy} | Selected={w.worker_id}")
            return w

    def _inc_active(self, wid): self.active_requests[wid] = self.active_requests.get(wid, 0) + 1
    def _dec_active(self, wid):
        if wid in self.active_requests: self.active_requests[wid] = max(0, self.active_requests[wid] - 1)

    async def _record_failure(self, wid):
        async with self._state_lock:
            self.worker_health[wid]["failures"] += 1
            if self.worker_health[wid]["failures"] >= 3: self.worker_health[wid]["status"] = "degraded"
        self._dec_active(wid)

    async def _record_success(self, wid):
        async with self._state_lock:
            self.worker_health[wid]["failures"] = 0
            self.worker_health[wid]["status"] = "healthy"
        self._dec_active(wid)

    async def process_request(self, req: TaskRequest) -> dict:
        self.metrics["assigned"] += 1
        attempt, err = 0, None
        while attempt < self.max_retries:
            w = self._get_next_worker()
            async with self.semaphore:
                self._inc_active(w.worker_id)
                logger.info(f"[TASK] Attempt {attempt+1}/{self.max_retries} | Worker={w.worker_id} | Queue={self.active_requests[w.worker_id]}")
                try:
                    if random.random() < self.fault_rate: raise ConnectionError(f"Simulated crash: {w.worker_id}")
                    
                    prompt = req.prompt
                    if self.use_rag and req.use_rag and hasattr(self, 'rag'):
                        #  FIX #1: Run blocking RAG query in thread pool to avoid freezing event loop
                        ctx = await asyncio.to_thread(self.rag.query, req.prompt, n_results=2)
                        if ctx:
                            prompt = f"Context:\n{ctx}\n\nQuestion: {req.prompt}"
                            logger.info(f"[RAG] Context injected ({len(ctx)} chars) | Prompt enhanced")
                    
                    task = InferenceRequest(prompt=prompt, max_tokens=req.max_tokens)
                    
                    # FIX #3: Add timeout to prevent worker hangs from blocking semaphore
                    try:
                        res = await asyncio.wait_for(w.process_task(task), timeout=300.0)
                    except asyncio.TimeoutError:
                        raise ConnectionError(f"Worker {w.worker_id} timed out after 300s")
                    
                    if res.get("status") == "success":
                        async with self._state_lock:
                            self.metrics["completed"] += 1
                            tot = self.metrics["completed"]
                            prev = self.metrics["avg_lat"]
                            self.metrics["avg_lat"] = prev + ((res["latency_ms"] - prev) / tot)
                        await self._record_success(w.worker_id)
                        res["use_rag"] = req.use_rag; res["routing"] = self.routing_strategy
                        logger.info(f"[TASK] SUCCESS | Worker={w.worker_id} | Latency={res['latency_ms']:.0f}ms | Tokens={res.get('tokens_generated',0)}")
                        if attempt > 0:
                            async with self._state_lock: self.metrics["reassigned"] += 1
                            logger.info(f"[FAULT] RECOVERY | Task reassigned & completed on {w.worker_id}")
                        return res
                except Exception as e:
                    err = str(e)
                    await self._record_failure(w.worker_id)
                    logger.warning(f"[FAULT] FAILURE | Worker={w.worker_id} | Error: {err}")
            attempt += 1
        async with self._state_lock: self.metrics["failed"] += 1
        logger.error(f"[FAULT] PERMANENT DROP | Task failed after {self.max_retries} retries | Error: {err}")
        return {"status": "error", "message": err or "Worker failure", "worker_id": "unknown"}

    def get_status(self):
        return {**self.metrics, "worker_health": {k: {**v, "age": time.time()-v["hb_time"]} for k,v in self.worker_health.items()},
                "active": self.active_requests, "routing": self.routing_strategy,
                "hb_active": self._hb_started and self._hb_task and not self._hb_task.done()}

    async def shutdown(self):
        if self._hb_task: self._hb_task.cancel(); await asyncio.gather(self._hb_task, return_exceptions=True)
        for w in self.workers: await w.close()
        logger.info("[SHUTDOWN] Scheduler stopped")
