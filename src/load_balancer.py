import os, sys, asyncio, logging, time, uuid, json, argparse
from pathlib import Path
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

WORKER_ENDPOINTS = [
    os.getenv(f"WORKER_{i+1}_URL", f"http://localhost:{8081+i}")
    for i in range(int(os.getenv("NUM_WORKERS", "4")))
]

scheduler = MasterScheduler(
    worker_endpoints=WORKER_ENDPOINTS,
    model_name=os.getenv("MODEL_NAME", "phi3:mini"),
    routing_strategy=os.getenv("ROUTING_STRATEGY", "load_aware"),
    use_rag=True,
    fault_rate=float(os.getenv("FAULT_RATE", "0.0")),
    max_retries=int(os.getenv("MAX_RETRIES", "2"))
)

# Leader Election State File
LEADER_FILE = Path(__file__).resolve().parent.parent / "logs" / "leader.json"
LEADER_FILE.parent.mkdir(parents=True, exist_ok=True)

def _write_leader(role: str, port: int, status: str = "active"):
    data = {"role": role, "port": port, "status": status, "timestamp": time.time(), "pid": os.getpid()}
    LEADER_FILE.write_text(json.dumps(data))

async def _primary_heartbeat(port: int):
    """Primary LB writes heartbeat every 3 seconds."""
    while True:
        _write_leader("primary", port)
        await asyncio.sleep(3)

async def _standby_monitor(primary_port: int, standby_port: int):
    """Standby LB promotes itself if primary misses 3 heartbeats (~10s)."""
    consecutive_misses = 0
    promoted = False
    while not promoted:
        await asyncio.sleep(3)
        try:
            if not LEADER_FILE.exists():
                continue
            data = json.loads(LEADER_FILE.read_text())
            if time.time() - data.get("timestamp", 0) > 10:
                consecutive_misses += 1
                logger.warning(f"[FAILOVER] Primary unresponsive ({consecutive_misses}/3)")
                if consecutive_misses >= 3:
                    logger.info(f"[FAILOVER] Promoting standby to ACTIVE (port {standby_port})")
                    _write_leader("standby_promoted", standby_port, "active")
                    promoted = True
                    break
            else:
                consecutive_misses = 0
        except Exception as e:
            logger.error(f"[FAILOVER] Monitor error: {e}")

async def handle_infer(request: web.Request):
    rid = str(uuid.uuid4())[:8]
    start = time.time()
    try:
        body = await request.json()
        req = InferReq(**body)
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e), "request_id": rid}, status=400)
    try:
        task = TaskRequest(prompt=req.prompt, max_tokens=req.max_tokens, priority=5, use_rag=req.use_rag)
        res = await scheduler.process_request(task)
        return web.json_response({**res, "request_id": rid, "latency_ms": round((time.time()-start)*1000, 2)})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e), "request_id": rid}, status=500)

async def handle_health(request):
    status = scheduler.get_status()
    worker_health = status.get("worker_health", {})
    workers_healthy = sum(1 for h in worker_health.values() if isinstance(h, dict) and h.get("status") == "healthy")
    scheduler_active = status.get("hb_active", False)
    
    # Load leader state for failover visibility
    leader_data = {}
    if LEADER_FILE.exists():
        try: leader_data = json.loads(LEADER_FILE.read_text())
        except: pass
        
    return web.json_response({
        "status": "healthy" if (scheduler_active and workers_healthy > 0) else "starting",
        "service": "load-balancer",
        "scheduler_active": scheduler_active,
        "workers_healthy": workers_healthy,
        "total_workers": len(worker_health),
        "role": LEADER_FILE.exists() and json.loads(LEADER_FILE.read_text()).get("role", "unknown") or "unknown",
        "leader_state": leader_data,
        "timestamp": time.time()
    })

def create_app():
    app = web.Application()
    app.add_routes([web.post('/infer', handle_infer), web.get('/health', handle_health)])
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["primary", "standby"], default="primary")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--primary-port", type=int, default=8080, help="Port to monitor for failover")
    args = parser.parse_args()
    
    logger.info(f"Starting LB [{args.role.upper()}] on port {args.port}")
    _write_leader(args.role, args.port, "starting")
    
    app = create_app()
    
    # Run background tasks based on role
    async def run_lb():
        if args.role == "primary":
            asyncio.create_task(_primary_heartbeat(args.port))
            logger.info("Primary heartbeat active. Standby will monitor this port.")
        else:
            asyncio.create_task(_standby_monitor(args.primary_port, args.port))
            logger.info(f"Standby active. Monitoring primary on port {args.primary_port}.")
        
        await web._run_app(app, host="0.0.0.0", port=args.port, print=None)
    
    try:
        asyncio.run(run_lb())
    except KeyboardInterrupt:
        logger.info("LB shutdown by user")