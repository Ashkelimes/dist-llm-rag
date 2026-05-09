import os, sys, asyncio, aiohttp, time, logging, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from logging_config import setup_component_logger
from metrics_logger import MetricsLogger

logger = setup_component_logger("test")
BASE = "http://localhost:8080/infer"

async def send_request(session, idx, use_rag=False):
    """Send a single inference request and return structured result."""
    start = time.time()
    payload = {"prompt": f"Test {idx}: Explain distributed caching.", "max_tokens": 10}
    if use_rag:
        payload["use_rag"] = True
    
    try:
        async with session.post(BASE, json=payload) as r:
            data = await r.json() if r.content_type == "application/json" else {}
            latency = (time.time() - start) * 1000
            success = (r.status == 200 and data.get("status") == "success")
            return {
                "req_id": idx,
                "status": r.status,
                "latency_ms": round(latency, 2),
                "tokens": data.get("tokens_generated", 0),
                "worker": data.get("worker_id", ""),
                "success": success,
                "gpu_util": data.get("gpu_util_percent"),
                "use_rag": data.get("use_rag", False),
                "error": None if success else data.get("message")
            }
    except asyncio.CancelledError:
        logger.warning(f"Request {idx} cancelled")
        return {
            "req_id": idx,
            "success": False,
            "error": "Cancelled",
            "latency_ms": 0,
            "tokens": 0
        }
    except asyncio.TimeoutError:
        logger.error(f"Request {idx} timed out")
        return {
            "req_id": idx,
            "success": False,
            "error": "Timeout",
            "latency_ms": 0,
            "tokens": 0
        }
    except Exception as e:
        logger.error(f"Request {idx} failed: {type(e).__name__}: {e}")
        return {
            "req_id": idx,
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "latency_ms": 0,
            "tokens": 0
        }

async def run_test(count: int, use_rag: bool = False):
    """Execute concurrent load test with metrics collection."""
    run_id = os.getenv("RUN_ID", "unknown")
    rag_suffix = "_rag" if use_rag else ""
    mlog = MetricsLogger(run_id=f"{run_id}_concurrent_{count}{rag_suffix}")
    
    # Concurrency limit: match worker count to avoid Ollama queue saturation
    concurrency = 4
    timeout_val = 1800 if use_rag else 1200
    
    connector = aiohttp.TCPConnector(limit=concurrency * 2, force_close=True)
    timeout = aiohttp.ClientTimeout(total=timeout_val, connect=10, sock_read=timeout_val)
    
    rag_note = " [RAG ENABLED]" if use_rag else ""
    logger.info(f"Starting test: {count} requests{rag_note} | Concurrency: {concurrency} | Timeout: {timeout_val}s")
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
        sema = asyncio.Semaphore(concurrency)
        async def run(idx):
            async with sema:
                return await send_request(sess, idx, use_rag)
        results = await asyncio.gather(*[run(i) for i in range(count)])
    
    # Log all results to metrics collector
    for r in results:
        if r.get("req_id") is not None:
            mlog.log_inference(
                r["req_id"],
                r.get("latency_ms", 0),
                r.get("tokens", 0),
                r.get("worker", "unknown"),
                "least_connections",
                r.get("success", False),
                r.get("error"),
                use_rag=r.get("use_rag", False),
                gpu_util=r.get("gpu_util")
            )
    mlog.flush()
    
    # Compute summary statistics
    ok = sum(1 for r in results if r.get("success"))
    lats = [r["latency_ms"] for r in results if r.get("success") and r.get("latency_ms", 0) > 0]
    avg_lat = sum(lats) / len(lats) if lats else 0
    total_tokens = sum(r.get("tokens", 0) for r in results)
    
    # Console output for immediate feedback
    print(f"\n{'='*60}")
    print(f"CONCURRENT TEST RESULTS{rag_note}: {ok}/{count} successful ({ok/max(count,1)*100:.1f}%)")
    print(f"{'='*60}")
    if lats:
        print(f"Avg Latency: {avg_lat:.2f}ms | Min: {min(lats):.0f}ms | Max: {max(lats):.0f}ms")
    print(f"Total Tokens: {total_tokens}")
    print(f"Metrics saved: {mlog.csv_path}")
    print(f"Summary: {mlog.summary_path}")
    
    # Logger output for file-based audit trail
    logger.info(f"Test Complete{rag_note}: {ok}/{count} ({ok/max(count,1)*100:.1f}%)")
    if lats:
        logger.info(f"Avg Latency: {avg_lat:.2f}ms | Tokens: {total_tokens}")
    logger.info(f"Metrics: {mlog.csv_path}")
    
    return 0 if ok == count else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent load test for distributed LLM system")
    parser.add_argument("--count", type=int, default=10, help="Number of concurrent requests (10-1000)")
    parser.add_argument("--use-rag", action="store_true", help="Enable RAG context injection")
    args = parser.parse_args()
    
    if not (10 <= args.count <= 1000):
        print("Error: --count must be between 10 and 1000")
        sys.exit(1)
    
    if args.use_rag and args.count > 10:
        confirm = input(f"RAG + {args.count} requests may timeout. Continue? (y/n): ").strip().lower()
        if confirm != "y":
            print("Test cancelled.")
            sys.exit(0)
    
    exit_code = asyncio.run(run_test(args.count, use_rag=args.use_rag))
    sys.exit(exit_code)