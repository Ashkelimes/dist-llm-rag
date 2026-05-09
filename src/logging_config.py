import logging
import os
from pathlib import Path
from datetime import datetime

RUN_ID = os.getenv("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
LOG_DIR = Path("logs/runs") / RUN_ID
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_component_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def get_run_id() -> str: return RUN_ID
def get_log_dir() -> Path: return LOG_DIR
