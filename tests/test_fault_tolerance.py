import os, sys, asyncio, time, logging
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from logging_config import setup_component_logger
from metrics_logger import MetricsLogger
from scheduler import MasterScheduler, TaskRequest

logger = setup_component_logger("fault_test")
RUN_ID = os.getenv("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))

async def run_fault_test(count: int = 200, rate: float = 0.25):
    mlog = MetricsLogger(run_id=f"{RUN_ID}_fault_{count}")
    sched = MasterScheduler(num_workers=4, max_retries=2, fault_rate=rate)
    await sched.start_health_monitor()
    
    start = time.time()
    # FIX #1: Add semaphore to limit concurrent scheduler calls
    sema = asyncio.Semaphore(50)
    
    async def task(i):
        async with sema:
            try:
                # FIX #2: Add timeout to prevent hangs
                res = await asyncio.wait_for(sched.process_request(TaskRequest(prompt=f"Fault {i}: Load balancing?", max_tokens=20)), timeout=350.0)
            except asyncio.TimeoutError:
                logger.error(f"Request {i} timed out after 350s")
                res = {"status": "error", "message": "Scheduler timeout", "worker_id": "unknown", "latency_ms": 350000, "tokens_generated": 0}
            mlog.log_inference(i, res.get("latency_ms",0), res.get("tokens_generated",0),
                            res.get("worker_id",""), "least_connections", res.get("status")=="success", res.get("message"))
            return res
    
    results = await asyncio.gather(*[task(i) for i in range(count)])
    mlog.flush()
    
    ok = sum(1 for r in results if r.get("status")=="success")
    elapsed = time.time() - start
    
    logger.info(f"Fault test complete: Success={ok}/{count} ({ok/count*100:.1f}%) | Time: {elapsed:.0f}s")
    logger.info(f"Reassigned: {sched.metrics['reassigned']} | Drops: {sched.metrics['failed']}")
    logger.info(f"Metrics saved: {mlog.csv_path}")
    
    print(f"\nFault Results ({count} reqs, {rate*100}% fail rate): Success={ok}/{count} ({ok/count*100:.1f}%)")
    print(f"Reassigned: {sched.metrics['reassigned']} | Drops: {sched.metrics['failed']}")
    print(f"Metrics saved: {mlog.csv_path}")

if __name__ == "__main__": 
    asyncio.run(run_fault_test())
