import os, sys, asyncio, time, csv
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scheduler import MasterScheduler, TaskRequest

LOG_FILE = "logs/fault_tolerance_run.csv"

async def run_fault_test(count: int = 200, fault_rate: float = 0.25):
    print(f"Starting Fault Tolerance Test: {count} requests | Simulated Failure Rate: {fault_rate*100:.0f}%")
    sched = MasterScheduler(num_workers=4, max_retries=2, fault_injection_rate=fault_rate)
    start = time.time()
    
    results = []
    async def task(i):
        req = TaskRequest(prompt=f"Fault test {i}: What is load balancing?", max_tokens=20)
        result = await sched.process_request(req)
        return {
            "request_id": i,
            "status": result.get("status", "unknown"),
            "worker_id": result.get("worker_id", "unknown"),
            "latency_ms": result.get("latency_ms", 0),
            "tokens": result.get("tokens_generated", 0),
            "error": result.get("message", ""),
            "reassigned": result.get("latency_ms", 0) > 0 and "error" in result.get("message", ""),
            "timestamp": datetime.now().isoformat()
        }
        
    results = await asyncio.gather(*[task(i) for i in range(count)])
    elapsed = time.time() - start
    
    # Save to CSV
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id","status","worker_id","latency_ms","tokens","error","reassigned","timestamp"])
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    reassigned = sum(1 for r in results if r["reassigned"])
    permanent_drops = sum(1 for r in results if r["status"] == "error")
    
    print(f"\nFault Tolerance Results ({count} requests):")
    print(f"   Success Rate: {successful}/{count} ({successful/count*100:.1f}%)")
    print(f"   Simulated Failures Triggered: ~{int(count*fault_rate)}")
    print(f"   Auto-Reassignments: {reassigned}")
    print(f"   Permanent Drops: {permanent_drops}")
    print(f"   Total Time: {elapsed:.2f}s | Avg Latency: {sched.metrics['avg_latency_ms']:.2f}ms")
    print(f"   Worker Health: { {k: v['status'] for k,v in sched.worker_health.items()} }")
    print(f"   CSV log: {LOG_FILE}")
    print(f"   Event log: logs/scheduler.log (search for RECOVERY/TASK_FAILED)")

if __name__ == "__main__":
    asyncio.run(run_fault_test())
