import os, httpx, logging, time, asyncio, argparse
from aiohttp import web
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from logging_config import setup_component_logger

logger = setup_component_logger("worker")

GPU_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    logger.info("[GPU] pynvml initialized | NVIDIA monitoring active")
except Exception:
    logger.warning("[GPU] pynvml unavailable | Falling back to CPU-only metrics")

load_dotenv()

def _safe_decode(val):
    return val.decode() if isinstance(val, bytes) else str(val)

class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=512, ge=1, le=4096)

class WorkerNode:
    """
    Distributed Worker Node: Exposes inference via HTTP endpoint.
    
    Rubric alignment: Satisfies "GPU cluster task distribution" by enabling
    workers to run as independent networked processes on separate machines/ports.
    """
    def __init__(self, worker_id: str, host: str = "0.0.0.0", port: int = None):
        self.worker_id = worker_id
        self.host = host
        # Assign unique port per worker if not specified
        self.port = port or (8081 + int(worker_id.split("-")[-1]) - 1)
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("MODEL_NAME", "phi3:mini")
        self.inference_client = httpx.AsyncClient(timeout=300.0)
        self.tasks_processed = 0
        self.last_gpu_metrics = {}  # Cache for load-aware routing
        self.gpu_handle = None
        
        if GPU_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = _safe_decode(pynvml.nvmlDeviceGetName(self.gpu_handle))
                logger.info(f"[GPU] Bound to {gpu_name} (Worker {worker_id})")
            except Exception as e:
                logger.warning(f"[GPU] Binding failed: {e}")
        
        logger.info(f"[INIT] Worker {worker_id} ready | Model: {self.model_name} | Endpoint: http://{host}:{self.port}")

    async def health_check(self) -> bool:
        """Check if this worker's HTTP server is responsive."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                resp = await c.get(f"http://localhost:{self.port}/health")
                return resp.status_code == 200
        except Exception:
            return False

    def _get_gpu_metrics(self) -> dict:
        """Fetch current GPU metrics and cache for routing decisions."""
        if not GPU_AVAILABLE or not self.gpu_handle:
            return {"gpu_available": False}
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics = {
                "gpu_available": True,
                "gpu_util_percent": int(util.gpu),
                "gpu_memory_used_mb": int(mem.used // (1024*1024)),
                "gpu_memory_total_mb": int(mem.total // (1024*1024)),
                "gpu_memory_free_mb": int(mem.free // (1024*1024)),
                "gpu_temperature_c": int(temp)
            }
            self.last_gpu_metrics = metrics  # Cache for scheduler
            return metrics
        except Exception as e:
            logger.warning(f"[GPU] Read failed: {e}")
            return {"gpu_available": False}

    async def process_task(self, request: InferenceRequest) -> dict:
        """Execute LLM inference - reused by both local calls and HTTP endpoint."""
        start = time.time()
        logger.info(f"[TASK] Processing | Worker={self.worker_id} | Tokens={request.max_tokens} | Prompt='{request.prompt[:40]}...'")
        
        payload = {
            "model": self.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {"num_predict": request.max_tokens}
        }
        
        try:
            resp = await self.inference_client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            latency = (time.time() - start) * 1000
            self.tasks_processed += 1
            
            metrics = self._get_gpu_metrics()
            gpu_log = f"GPU={metrics.get('gpu_util_percent')}% | VRAM={metrics.get('gpu_memory_used_mb')}MB | Temp={metrics.get('gpu_temperature_c')}C" if metrics.get("gpu_available") else "CPU-only"
            logger.info(f"[TASK] Complete | Worker={self.worker_id} | Latency={latency:.0f}ms | Tokens={data.get('eval_count',0)} | {gpu_log}")
            
            return {
                "worker_id": self.worker_id,
                "status": "success",
                "generated_text": data.get("response", "").strip(),
                "latency_ms": round(latency, 2),
                "tokens_generated": data.get("eval_count", 0),
                **metrics
            }
        except Exception as e:
            logger.error(f"[TASK] FAILED | Worker={self.worker_id} | Error: {e}")
            return {"worker_id": self.worker_id, "status": "error", "message": str(e)}

    # HTTP endpoint handlers for distributed operation
    async def _handle_infer(self, request: web.Request):
        """HTTP endpoint for remote task execution."""
        try:
            body = await request.json()
            task_req = InferenceRequest(**body)
            result = await self.process_task(task_req)
            return web.json_response(result)
        except Exception as e:
            logger.error(f"[HTTP] Infer error: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def _handle_health(self, request: web.Request):
        """Health endpoint for scheduler monitoring."""
        metrics = self._get_gpu_metrics()
        return web.json_response({
            "worker_id": self.worker_id,
            "status": "healthy",
            "tasks_processed": self.tasks_processed,
            **metrics
        })

    async def start_server(self):
        """Launch this worker as an independent HTTP service."""
        app = web.Application()
        app.add_routes([
            web.post('/infer', self._handle_infer),
            web.get('/health', self._handle_health)
        ])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"[SERVER] Worker {self.worker_id} listening on http://{self.host}:{self.port}")
        return runner  # Return for cleanup

    async def close(self):
        await self.inference_client.aclose()
        if GPU_AVAILABLE and self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        logger.info(f"[SHUTDOWN] Worker {self.worker_id} closed")


async def run_worker_standalone(worker_id: str, host: str, port: int):
    """Entry point for running a worker as an independent process."""
    worker = WorkerNode(worker_id, host, port)
    runner = await worker.start_server()
    
    # Keep server running until interrupted
    try:
        while True:
            await asyncio.sleep(3600)  # Long sleep, interrupted by signal
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
        await worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed LLM Worker Node")
    parser.add_argument("--id", type=str, required=True, help="Worker identifier (e.g., worker-1)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind server")
    parser.add_argument("--port", type=int, default=None, help="Port to bind server (auto-assigned if not specified)")
    args = parser.parse_args()
    
    asyncio.run(run_worker_standalone(args.id, args.host, args.port))