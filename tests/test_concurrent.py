import os, sys, asyncio, aiohttp, time, csv, psutil, argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
BASE_URL = "http://localhost:8080/infer"
LOG_FILE = "logs/performance_run.csv"

async def send_request(session, req_id, prompt):
    max_retries = 2
    for attempt in range(max_retries):
        start = time.time()
        try:
            async with session.post(BASE_URL, json={"prompt": prompt, "max_tokens": 50}) as resp:
                data = await resp.json()
                latency = (time.time() - start) * 1000
                if resp.status == 200 and data.get("status") == "success":
                    return {"request_id": req_id, "status": resp.status, "latency_ms": round(latency, 2),
                            "tokens": data.get("tokens_generated", 0), "response_size": len(str(data)), "success": True, "error": ""}
                # Server-side 5xx or non-success: retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.5)
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1.5)
            else:
                return {"request_id": req_id, "status": 0, "latency_ms": round((time.time()-start)*1000, 2),
                        "tokens": 0, "response_size": 0, "success": False, "error": "Max retries exceeded"}

async def run_test(count: int):
    print(f" Starting concurrency test: {count} requests")
    process = psutil.Process()
    cpu_before = process.cpu_percent(interval=0.1)
    ram_before = process.memory_info().rss / 1024 / 1024
    
    # CRITICAL: Prevent client-side connection pool exhaustion
    connector = aiohttp.TCPConnector(limit=count, force_close=True)
    timeout = aiohttp.ClientTimeout(total=180, connect=10, sock_read=150)
    
    results = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Throttle active connections to avoid client-side DOS
        semaphore = asyncio.Semaphore(50)
        async def throttled(idx, prompt):
            async with semaphore:
                return await send_request(session, idx, prompt)
        tasks = [throttled(i, f"Test prompt {i}: Explain caching in distributed systems.") for i in range(count)]
        results = await asyncio.gather(*tasks)
        
    cpu_after = process.cpu_percent(interval=0.1)
    ram_after = process.memory_info().rss / 1024 / 1024
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id","status","latency_ms","tokens","response_size","success","error"])
        writer.writeheader()
        writer.writerows(results)
        
    successful = sum(1 for r in results if r["success"])
    avg_latency = sum(r["latency_ms"] for r in results if r["success"]) / max(successful, 1)
    total_tokens = sum(r["tokens"] for r in results)
    total_bytes = sum(r["response_size"] for r in results)
    
    print(f"\nResults ({count} requests):")
    print(f"   Success Rate: {successful}/{count} ({successful/count*100:.1f}%)")
    print(f"   Avg Latency: {avg_latency:.2f}ms")
    print(f"   Total Tokens: {total_tokens}")
    print(f"   Total Response Size: {total_bytes/1024:.2f} KB")
    print(f"   CPU ?: {cpu_after - cpu_before:+.1f}% | RAM ?: {ram_after - ram_before:+.1f} MB")
    print(f"   Full log: {LOG_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of concurrent requests")
    args = parser.parse_args()
    asyncio.run(run_test(args.count))
