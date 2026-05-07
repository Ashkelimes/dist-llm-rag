import os, sys, asyncio, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scheduler import MasterScheduler, TaskRequest

async def run_fault_test(count: int = 200, fault_rate: float = 0.25):
    print(f"???  Starting Fault Tolerance Test: {count} requests | Simulated Failure Rate: {fault_rate*100:.0f}%")
    sched = MasterScheduler(num_workers=4, max_retries=2, fault_injection_rate=fault_rate)
    start = time.time()
    
    async def task(i):
        req = TaskRequest(prompt=f"Fault test {i}: What is load balancing?", max_tokens=20)
        return await sched.process_request(req)
        
    results = await asyncio.gather(*[task(i) for i in range(count)])
    elapsed = time.time() - start
    
    successful = sum(1 for r in results if r.get("status") == "success")
    reassigned = sum(1 for r in results if r.get("latency_ms", 0) > 0 and "error" not in r)
    
    print(f"\n Fault Tolerance Results ({count} requests):")
    print(f"   Success Rate: {successful}/{count} ({successful/count*100:.1f}%)")
    print(f"   Simulated Failures Triggered: ~{int(count*fault_rate)}")
    print(f"   Auto-Reassignments: {sched.metrics['reassigned']}")
    print(f"   Permanent Drops: {sched.metrics['total_failed']}")
    print(f"   Total Time: {elapsed:.2f}s | Avg Latency: {sched.metrics['avg_latency_ms']:.2f}ms")
    print(f"   Worker Health: { {k: v['status'] for k,v in sched.worker_health.items()} }")
    print("\n?? Check logs/scheduler.log for 'RECOVERY' and 'TASK_FAILED' events.")

if __name__ == "__main__":
    asyncio.run(run_fault_test())
