import logging
import os
from pathlib import Path
from datetime import datetime

# Consistent RUN_ID across all processes (inherited from main.py env)
RUN_ID = os.getenv("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))

# Resolve to absolute path to prevent CWD discrepancies across subprocesses
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = (PROJECT_ROOT / "logs" / "runs" / RUN_ID).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_component_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if module is re-imported in same process
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    
    # File handler
    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    logger.info(f"Logger '{name}' initialized | Run: {RUN_ID} | Dir: {LOG_DIR}")
    return logger

def get_run_id() -> str: return RUN_ID
def get_log_dir() -> Path: return LOG_DIR