import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

class MetricsLogger:
    def __init__(self, run_id: str | None = None, log_dir: str = "logs/metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"metrics_{self.run_id}.csv"
        self.json_path = self.log_dir / f"metrics_{self.run_id}.json"
        self.summary_path = self.log_dir / f"summary_{self.run_id}.txt"
        self._buffer: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("metrics")
        self.logger.info(f"MetricsLogger initialized | Run ID: {self.run_id}")

    def log_event(self, event_type: str, **kwargs):
        entry = {"timestamp": datetime.now().isoformat(), "run_id": self.run_id, "event_type": event_type, **kwargs}
        self._buffer.append(entry)
        self.logger.info(f"{event_type}: {json.dumps(kwargs, default=str)}")

    def log_inference(self, request_id: str, latency_ms: float, tokens: int,
                     worker_id: str, routing_strategy: str, success: bool = True, error: str | None = None, **kwargs):
        self.log_event("inference", request_id=request_id, latency_ms=latency_ms,
                      tokens=tokens, worker_id=worker_id, routing_strategy=routing_strategy,
                      success=success, error=error, **kwargs)

    def log_heartbeat(self, worker_id: str, healthy: bool, response_time_ms: float | None = None):
        self.log_event("heartbeat", worker_id=worker_id, healthy=healthy, response_time_ms=response_time_ms)

    def log_routing(self, request_id: str, selected_worker: str, strategy: str, active_requests: Dict):
        self.log_event("routing", request_id=request_id, selected_worker=selected_worker,
                      strategy=strategy, active_requests=active_requests)

    def flush(self):
        if not self._buffer: return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._buffer[0].keys())
            writer.writeheader()
            writer.writerows(self._buffer)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "events": self._buffer}, f, indent=2, default=str)
        self._generate_summary()
        self.logger.info(f"Flushed {len(self._buffer)} events")
        self._buffer.clear()

    def _generate_summary(self):
        inferences = [e for e in self._buffer if e["event_type"] == "inference"]
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(f"METRICS SUMMARY | Run ID: {self.run_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            if inferences:
                success_list = [i for i in inferences if i.get("success", True)]
                f.write(f"Total Requests: {len(inferences)}\n")
                f.write(f"Successful: {len(success_list)} ({len(success_list)/max(len(inferences),1)*100:.1f}%)\n")
                if success_list:
                    lats = [i["latency_ms"] for i in success_list]
                    toks = [i["tokens"] for i in success_list]
                    f.write(f"Avg Latency: {sum(lats)/len(lats):.2f} ms\n")
                    f.write(f"Min/Max Latency: {min(lats):.2f} / {max(lats):.2f} ms\n")
                    f.write(f"Total Tokens: {sum(toks)}\n")
                gpu = [e for e in self._buffer if e.get("gpu_util_percent") is not None]
                if gpu:
                    f.write(f"\nGPU Avg Utilization: {sum(g['gpu_util_percent'] for g in gpu)/len(gpu):.1f}%\n")
                    f.write(f"GPU Avg Temperature: {sum(g['gpu_temperature_c'] for g in gpu)/len(gpu):.1f}C\n")
            f.write("\n" + "="*60 + f"\nFiles: {self.csv_path.name}, {self.summary_path.name}\n")
