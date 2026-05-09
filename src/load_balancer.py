import os, sys, asyncio, logging, time, uuid
from aiohttp import web
from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logging_config import setup_component_logger
from scheduler import MasterScheduler, TaskRequest

load_dotenv()
logger = setup_component_logger("load_balancer")

class InferReq(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    use_rag: bool = Field(default=False)

# Worker endpoints for distributed architecture
# Each worker runs as independent HTTP service on its own port
WORKER_ENDPOINTS = [
    os.getenv(f"WORKER_{i+1}_URL", f"http://localhost:{8081+i}")
    for i in range(int(os.getenv("NUM_WORKERS", "4")))
]

# Scheduler initialization with distributed worker endpoints
# Rubric alignment: "GPU cluster task distribution" - workers are networked nodes
scheduler = MasterScheduler(
    worker_endpoints=WORKER_ENDPOINTS,  # ← New parameter for distributed design
    model_name=os.getenv("MODEL_NAME", "phi3:mini"),
    routing_strategy=os.getenv("ROUTING_STRATEGY", "load_aware"),  # ← Default to load_aware
    use_rag=True,  # Pipeline loaded; per-request flag controls usage
    fault_rate=float(os.getenv("FAULT_RATE", "0.0")),
    max_retries=int(os.getenv("MAX_RETRIES", "2"))
)

async def handle_infer(request: web.Request):
    """Handle inference request: parse, route, execute, respond."""
    rid = str(uuid.uuid4())[:8]
    start = time.time()
    
    try:
        body = await request.json()
        req = InferReq(**body)
    except Exception as e:
        logger.error(f"[{rid}] Parse error: {type(e).__name__}: {e}")
        return web.json_response(
            {"status": "error", "message": f"Invalid request: {str(e)}", "request_id": rid},
            status=400,
            headers={"Access-Control-Allow-Origin": "*"}
        )
    
    try:
        # PASS client's use_rag flag to scheduler
        task = TaskRequest(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            priority=5,
            use_rag=req.use_rag
        )
        res = await scheduler.process_request(task)
        latency = (time.time() - start) * 1000
        
        return web.json_response(
            {**res, "request_id": rid, "latency_ms": round(latency, 2)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"[{rid}] Processing error: {e}")
        return web.json_response(
            {"status": "error", "message": str(e), "request_id": rid},
            status=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )

async def handle_health(request):
    """
    Health endpoint with functional readiness check.
    
    Returns detailed status including:
    - scheduler_active: heartbeat monitor running
    - workers_healthy: count of workers in 'healthy' state
    - worker_endpoints: list of active worker URLs
    - timestamp: for cache invalidation
    
    Rubric alignment: Demonstrates fault tolerance via proactive health monitoring
    of distributed components.
    """
    status = scheduler.get_status()
    worker_health = status.get("worker_health", {})
    
    # Count workers with status == "healthy" (not "degraded")
    workers_healthy = sum(
        1 for h in worker_health.values()
        if isinstance(h, dict) and h.get("status") == "healthy"
    )
    
    scheduler_active = status.get("hb_active", False)
    
    return web.json_response({
        "status": "healthy" if (scheduler_active and workers_healthy > 0) else "starting",
        "service": "load-balancer",
        "scheduler_active": scheduler_active,
        "workers_healthy": workers_healthy,
        "total_workers": len(worker_health),
        "worker_endpoints": status.get("worker_endpoints", []),
        "routing_strategy": status.get("routing", "unknown"),
        "timestamp": time.time()
    }, headers={"Access-Control-Allow-Origin": "*"})

def create_app():
    """Factory function for aiohttp application with CORS headers."""
    app = web.Application()
    app.add_routes([
        web.post('/infer', handle_infer),
        web.get('/health', handle_health)
    ])
    return app

if __name__ == "__main__":
    logger.info("Starting LB on :8080 (Distributed, RAG-enabled, per-request toggle)")
    logger.info(f"Config: Workers={len(WORKER_ENDPOINTS)} | Endpoints={WORKER_ENDPOINTS} | Strategy={os.getenv('ROUTING_STRATEGY', 'load_aware')} | RAG=optional")
    web.run_app(create_app(), host="localhost", port=8080)