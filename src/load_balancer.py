import os
import sys
import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from aiohttp import web
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler import MasterScheduler, TaskRequest

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/load_balancer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InferRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=512, ge=1, le=4096)

class InferResponse(BaseModel):
    request_id: str
    status: str
    result: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    tokens_generated: int | None = None
    worker_id: str | None = None

# Global async scheduler (4 workers, Round Robin)
scheduler = MasterScheduler(num_workers=4, model_name=os.getenv("MODEL_NAME", "phi3:mini"))

async def handle_infer(request: web.Request) -> web.Response:
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received inference request")
    
    try:
        body = await request.json()
        req = InferRequest(**body)
    except ValidationError as e:
        return web.json_response(
            InferResponse(request_id=request_id, status="error", error=str(e)).model_dump(),
            status=400,
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception:
        return web.json_response(
            InferResponse(request_id=request_id, status="error", error="Invalid JSON").model_dump(),
            status=400,
            headers={"Access-Control-Allow-Origin": "*"}
        )
    
    try:
        task = TaskRequest(prompt=req.prompt, max_tokens=req.max_tokens, priority=5)
        result = await scheduler.process_request(task)
        
        latency = (time.time() - start_time) * 1000
        response = InferResponse(
            request_id=request_id,
            status=result.get("status", "error"),
            result=result.get("generated_text"),
            error=result.get("message"),
            latency_ms=round(latency, 2),
            tokens_generated=result.get("tokens_generated"),
            worker_id=result.get("worker_id")
        )
        logger.info(f"[{request_id}] Completed in {latency:.2f}ms via {result.get('worker_id')}")
        return web.json_response(response.model_dump(), headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Processing error: {e}")
        return web.json_response(
            InferResponse(request_id=request_id, status="error", error=str(e), latency_ms=round(latency, 2)).model_dump(),
            status=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )

async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "healthy", "service": "load-balancer"}, headers={"Access-Control-Allow-Origin": "*"})

async def handle_status(request: web.Request) -> web.Response:
    return web.json_response({
        "service": "load-balancer",
        "scheduler_metrics": scheduler.get_status(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, headers={"Access-Control-Allow-Origin": "*"})

def create_app() -> web.Application:
    app = web.Application()
    app.add_routes([
        web.post('/infer', handle_infer),
        web.get('/health', handle_health),
        web.get('/status', handle_status),
    ])
    logger.info("Load balancer routes configured (Async Multi-Worker)")
    return app

if __name__ == "__main__":
    logger.info("Starting load balancer on http://localhost:8080")
    app = create_app()
    web.run_app(app, host="localhost", port=8080, print=lambda x: logger.info(x))
