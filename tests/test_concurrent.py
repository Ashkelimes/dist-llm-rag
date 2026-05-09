import os, sys, asyncio, aiohttp, time, logging, argparse, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from logging_config import setup_component_logger
from metrics_logger import MetricsLogger

logger = setup_component_logger("test")
PRIMARY_BASE = os.getenv("LB_URL", "http://localhost:8080/infer")
FALLBACK_BASE = None

# Thread-safe failover state
_failover_lock = threading.Lock()
_faILOVER_ACTIVE = False
_faILOVER_TIMESTAMP = 0

def _should_use_fallback() -> bool:
    """Thread-safe check: use fallback if failover triggered within last 60s."""
    global _faILOVER_ACTIVE, _faILOVER_TIMESTAMP
    with _failover_lock:
        if _faILOVER_ACTIVE and (time.time() - _faILOVER_TIMESTAMP) < 60:
            return True
        return False

def _trigger_failover():
    """Thread-safe failover trigger with timestamp."""
    global _faILOVER_ACTIVE, _faILOVER_TIMESTAMP
    with _failover_lock:
        if not _faILOVER_ACTIVE:
            _faILOVER_ACTIVE = True
            _faILOVER_TIMESTAMP = time.time()
            logger.info(f"FAILOVER TRIGGERED at {time.strftime('%H:%M:%S')}")

async def send_request(session: aiohttp.ClientSession, idx: int, use_rag: bool = False):
    """Send request with automatic, thread-safe failover to standby."""
    # Determine target endpoint
    if _should_use_fallback() and FALLBACK_BASE:
        target = FALLBACK_BASE
    else:
        target = PRIMARY_BASE
    
    start = time.time()
    payload = {"prompt": f"Test {idx}: Explain distributed caching.", "max_tokens": 10}
    if use_rag:
        payload["use_rag"] = True
    
    # Primary attempt
    try:
        async with session.post(target, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as r:
            data = await r.json() if r.content_type == "application/json" else {}
            latency = (time.time() - start) * 1000
            success = (r.status == 200 and data.get("status") == "success")
            return {
                "req_id": idx, "status": r.status, "latency_ms": round(latency, 2),
                "tokens": data.get("tokens_generated", 0), "worker": data.get("worker_id", ""),
                "success": success, "gpu_util": data.get("gpu_util_percent"),
                "use_rag": data.get("use_rag", False), "error": None if success else data.get("message"),
                "endpoint": target
            }
    except (aiohttp.ClientConnectorError, aiohttp.ClientOSError, OSError, ConnectionRefusedError) as e:
        # PRIMARY failed -> trigger failover (thread-safe, only once)
        if not _should_use_fallback() and FALLBACK_BASE:
            logger.warning(f"Request {idx}: PRIMARY unreachable. Triggering failover to STANDBY...")
            _trigger_failover()
            # Brief pause to let standby complete promotion
            await asyncio.sleep(2.0)
            target = FALLBACK_BASE
        elif FALLBACK_BASE:
            target = FALLBACK_BASE
        else:
            # No fallback configured
            return {"req_id": idx, "success": False, "error": f"Connection failed: {e}", "latency_ms": 0, "tokens": 0, "endpoint": target}
        
        # Retry on fallback
        try:
            async with session.post(target, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as r:
                data = await r.json() if r.content_type == "application/json" else {}
                latency = (time.time() - start) * 1000
                success = (r.status == 200 and data.get("status") == "success")
                return {
                    "req_id": idx, "status": r.status, "latency_ms": round(latency, 2),
                    "tokens": data.get("tokens_generated", 0), "worker": data.get("worker_id", ""),
                    "success": success, "gpu_util": data.get("gpu_util_percent"),
                    "use_rag": data.get("use_rag", False), "error": None if success else data.get("message"),
                    "endpoint": target
                }
        except Exception as e2:
            logger.error(f"Request {idx} failed on fallback {target}: {type(e2).__name__}")
            return {"req_id": idx, "success": False, "error": f"Fallback failed: {e2}", "latency_ms": 0, "tokens": 0, "endpoint": target}
            
    except asyncio.CancelledError:
        return {"req_id": idx, "success": False, "error": "Cancelled", "latency_ms": 0, "tokens": 0, "endpoint": target}
    except asyncio.TimeoutError:
        return {"req_id": idx, "success": False, "error": "Timeout", "latency_ms": 0, "tokens": 0, "endpoint": target}
    except Exception as e:
        return {"req_id": idx, "success": False, "error": f"{type(e).__name__}: {e}", "latency_ms": 0, "tokens": 0, "endpoint": target}

async def run_test(count: int, use_rag: bool = False):
    """Execute concurrent load test with automatic, thread-safe failover."""
    run_id = os.getenv("RUN_ID", "unknown")
    rag_suffix = "_rag" if use_rag else ""
    mlog = MetricsLogger(run_id=f"{run_id}_concurrent_{count}{rag_suffix}")
    
    concurrency = 4
    timeout_val = 1800 if use_rag else 1200
    connector = aiohttp.TCPConnector(limit=concurrency * 2, force_close=True)
    timeout = aiohttp.ClientTimeout(total=timeout_val, connect=10, sock_read=timeout_val)
    
    rag_note = " [RAG ENABLED]" if use_rag else ""
    fallback_note = f" | Standby: {FALLBACK_BASE}" if FALLBACK_BASE else ""
    logger.info(f"Starting test: {count} requests{rag_note}{fallback_note} | Concurrency: {concurrency} | Timeout: {timeout_val}s")
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
        sema = asyncio.Semaphore(concurrency)
        async def run(idx):
            async with sema:
                return await send_request(sess, idx, use_rag)
        # Use return_exceptions=True to prevent one failure from cancelling all
        results = await asyncio.gather(*[run(i) for i in range(count)], return_exceptions=True)
    
    # Process results (handle exceptions from gather)
    processed = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"Request {i} raised exception: {r}")
            processed.append({"req_id": i, "success": False, "error": str(r), "latency_ms": 0, "tokens": 0, "endpoint": "unknown"})
        elif isinstance(r, dict):
            processed.append(r)
    
    # Log all results
    for r in processed:
        if r.get("req_id") is not None:
            mlog.log_inference(
                r["req_id"], r.get("latency_ms", 0), r.get("tokens", 0),
                r.get("worker", "unknown"), "load_aware",
                r.get("success", False), r.get("error"),
                use_rag=r.get("use_rag", False), gpu_util=r.get("gpu_util")
            )
    mlog.flush()
    
    # Summary stats
    ok = sum(1 for r in processed if r.get("success"))
    lats = [r["latency_ms"] for r in processed if r.get("success") and r.get("latency_ms", 0) > 0]
    avg_lat = sum(lats) / len(lats) if lats else 0
    total_tokens = sum(r.get("tokens", 0) for r in processed)
    standby_hits = sum(1 for r in processed if r.get("success") and r.get("endpoint") == FALLBACK_BASE)
    
    # Console output
    print(f"\n{'='*60}")
    print(f"CONCURRENT TEST RESULTS{rag_note}: {ok}/{count} successful ({ok/max(count,1)*100:.1f}%)")
    if standby_hits > 0:
        print(f"AUTOMATIC FAILOVER: {standby_hits} requests served by STANDBY")
    print(f"{'='*60}")
    if lats:
        print(f"Avg Latency: {avg_lat:.2f}ms | Min: {min(lats):.0f}ms | Max: {max(lats):.0f}ms")
    print(f"Total Tokens: {total_tokens}")
    print(f"Metrics saved: {mlog.csv_path}")
    print(f"Summary: {mlog.summary_path}")
    
    # Logger output
    logger.info(f"Test Complete{rag_note}: {ok}/{count} ({ok/max(count,1)*100:.1f}%)")
    if standby_hits > 0:
        logger.info(f"Automatic failover: {standby_hits} requests routed to standby")
    if lats:
        logger.info(f"Avg Latency: {avg_lat:.2f}ms | Tokens: {total_tokens}")
    logger.info(f"Metrics: {mlog.csv_path}")
    
    return 0 if ok == count else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent load test with automatic LB failover")
    parser.add_argument("--count", type=int, default=10, help="Number of concurrent requests (10-1000)")
    parser.add_argument("--use-rag", action="store_true", help="Enable RAG context injection")
    parser.add_argument("--fallback-port", type=int, default=None, help="Standby LB port for automatic failover")
    args = parser.parse_args()
    
    if not (10 <= args.count <= 1000):
        print("Error: --count must be between 10 and 1000")
        sys.exit(1)
    
    if args.fallback_port:
        FALLBACK_BASE = f"http://localhost:{args.fallback_port}/infer"
        logger.info(f"Failover enabled: PRIMARY={PRIMARY_BASE} | STANDBY={FALLBACK_BASE}")
    
    if args.use_rag and args.count > 10:
        confirm = input(f"RAG + {args.count} requests may timeout. Continue? (y/n): ").strip().lower()
        if confirm != "y":
            print("Test cancelled.")
            sys.exit(0)
    
    exit_code = asyncio.run(run_test(args.count, use_rag=args.use_rag))
    sys.exit(exit_code)