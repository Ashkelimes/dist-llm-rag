import os, httpx, logging, time, asyncio
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
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("MODEL_NAME", "phi3:mini")
        self.inference_client = httpx.AsyncClient(timeout=300.0)
        self.tasks_processed = 0
        self.gpu_handle = None
        if GPU_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = _safe_decode(pynvml.nvmlDeviceGetName(self.gpu_handle))
                logger.info(f"[GPU] Bound to {gpu_name} (Worker {worker_id})")
            except Exception as e:
                logger.warning(f"[GPU] Binding failed: {e}")
        logger.info(f"[INIT] Worker {worker_id} ready | Model: {self.model_name}")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                resp = await c.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def _get_gpu_metrics(self) -> dict:
        if not GPU_AVAILABLE or not self.gpu_handle:
            return {"gpu_available": False}
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            return {
                "gpu_available": True,
                "gpu_util_percent": int(util.gpu),
                "gpu_memory_used_mb": int(mem.used // (1024*1024)),
                "gpu_memory_total_mb": int(mem.total // (1024*1024)),
                "gpu_temperature_c": int(temp)
            }
        except Exception as e:
            logger.warning(f"[GPU] Read failed: {e}")
            return {"gpu_available": False}

    async def process_task(self, request: InferenceRequest) -> dict:
        start = time.time()
        logger.info(f"[TASK] Processing | Worker={self.worker_id} | Tokens={request.max_tokens} | Prompt='{request.prompt[:40]}...'")
        payload = {"model": self.model_name, "prompt": request.prompt, "stream": False, "options": {"num_predict": request.max_tokens}}
        try:
            resp = await self.inference_client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.time() - start) * 1000
            self.tasks_processed += 1
            
            metrics = self._get_gpu_metrics()
            gpu_log = f"GPU={metrics.get('gpu_util_percent')}% | VRAM={metrics.get('gpu_memory_used_mb')}MB | Temp={metrics.get('gpu_temperature_c')}C" if metrics.get("gpu_available") else "CPU-only"
            logger.info(f"[TASK] Complete | Worker={self.worker_id} | Latency={latency:.0f}ms | Tokens={data.get('eval_count',0)} | {gpu_log}")
            
            return {
                "worker_id": self.worker_id, "status": "success",
                "generated_text": data.get("response", "").strip(),
                "latency_ms": round(latency, 2), "tokens_generated": data.get("eval_count", 0),
                **metrics
            }
        except Exception as e:
            logger.error(f"[TASK] FAILED | Worker={self.worker_id} | Error: {e}")
            return {"worker_id": self.worker_id, "status": "error", "message": str(e)}

    async def close(self):
        await self.inference_client.aclose()
        if GPU_AVAILABLE and self.gpu_handle:
            try: pynvml.nvmlShutdown()
            except: pass
        logger.info(f"[SHUTDOWN] Worker {self.worker_id} closed")
