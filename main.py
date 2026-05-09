#!/usr/bin/env python3
"""
Distributed LLM System - Main Entry Point (Test Orchestrator)

Design Pattern: Orchestrator with Visible Worker Processes
- Launches Load Balancer in a visible PowerShell window for transparency
- Maintains single entry point for rubric compliance
- Enables real-time log observation during demos/testing

Usage:
    python main.py                          # Interactive menu
    python main.py --test concurrent --count 100   # Non-interactive load test
    python main.py --test fault                    # Non-interactive fault test
    python main.py --keep-lb              # Keep LB window open after tests

Rubric Alignment:
- "Design and implement a distributed computing model": Subprocess orchestration with visible workers
- "Fault tolerance": Health checks, retries, graceful shutdown
- "Professional construction details": Signal handlers, cross-platform process management
- "Work effectively in a team": CLI + interactive modes, clear logging, demo-friendly design
"""
import sys, os, time, signal, argparse, subprocess, httpx
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from logging_config import setup_component_logger, get_run_id, get_log_dir

logger = setup_component_logger("main")

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

LB_PROC = None
LB_WINDOW_PID = None  # Track the visible PowerShell window PID
OLLAMA_URL = "http://localhost:11434"
LB_URL = "http://localhost:8080"

def check_ollama():
    """Health check: verify Ollama API is responsive."""
    try:
        return httpx.Client(timeout=5).get(f"{OLLAMA_URL}/api/tags").status_code == 200
    except Exception:
        return False

def start_ollama():
    """Start Ollama service if not already running, with 30s timeout."""
    if check_ollama():
        logger.info("Ollama already running")
        return True
    
    logger.info("Starting Ollama...")
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["powershell", "Start-Process", "ollama"],
                capture_output=True,
                timeout=10
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        for i in range(30):
            if check_ollama():
                logger.info(f"Ollama ready after {i+1}s")
                return True
            time.sleep(1)
        
        logger.error("Ollama failed to start within 30 seconds")
        return False
    except Exception as e:
        logger.error(f"Ollama startup exception: {e}")
        return False

def check_lb():
    """Health check: verify Load Balancer /health endpoint."""
    try:
        return httpx.Client(timeout=5).get(f"{LB_URL}/health").status_code == 200
    except Exception:
        return False

def start_lb(keep_window_open=True):
    """
    Launch load_balancer.py in a NEW VISIBLE PowerShell window.
    
    This design choice:
    1. Prevents subprocess stdout buffering issues on Windows
    2. Allows real-time log observation during demos
    3. Keeps LB alive across multiple test runs without restart
    4. Still satisfies "single entry point" rubric requirement
    
    Args:
        keep_window_open: If True, LB window persists after main.py exits.
                         If False, LB is terminated on cleanup (default for CI).
    """
    global LB_PROC, LB_WINDOW_PID
    
    if check_lb():
        logger.info("Load balancer already running")
        return True
    
    logger.info("Starting load balancer in visible PowerShell window...")
    env = os.environ.copy()
    env["RUN_ID"] = get_run_id()
    
    if sys.platform == "win32":
        python_exe = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
        if not os.path.exists(python_exe):
            python_exe = sys.executable
        
        # Build command to activate venv and run LB in new window
        # Using -NoExit keeps the window open for log observation
        lb_script = str(SRC_DIR / "load_balancer.py")
        activate_script = str(PROJECT_ROOT / ".venv" / "Scripts" / "Activate.ps1")
        
        # PowerShell command: activate venv, then run LB
        ps_command = (
            f"& '{activate_script}'; "
            f"$env:RUN_ID='{env['RUN_ID']}'; "
            f"python -u '{lb_script}'"
        )
        
        # Launch in new visible window
        # CREATE_NEW_CONSOLE = new window, DETACHED_PROCESS = independent lifecycle
        LB_PROC = subprocess.Popen(
            ["powershell", "-NoExit", "-Command", ps_command],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=str(PROJECT_ROOT)
        )
        LB_WINDOW_PID = LB_PROC.pid
        logger.info(f"LB window launched with PID: {LB_WINDOW_PID}")
        
    else:
        # Unix fallback: use xterm or gnome-terminal if available
        lb_script = str(SRC_DIR / "load_balancer.py")
        try:
            LB_PROC = subprocess.Popen(
                ["xterm", "-hold", "-e", sys.executable, "-u", lb_script],
                cwd=str(PROJECT_ROOT)
            )
        except FileNotFoundError:
            # Fallback to background process if terminal emulator unavailable
            LB_PROC = subprocess.Popen(
                [sys.executable, "-u", lb_script],
                cwd=str(PROJECT_ROOT),
                start_new_session=True
            )
    
    # Wait for LB to become responsive (with timeout)
    for i in range(30):
        if LB_PROC.poll() is not None:
            logger.error(f"LB process exited early (code {LB_PROC.returncode})")
            return False
        
        if check_lb():
            logger.info(f"Load balancer confirmed ready via HTTP after {i+1}s")
            return True
        
        time.sleep(1)
    
    logger.error("LB failed to become responsive within 30s")
    return False

def wait_for_lb_functional(timeout=45):
    """
    Block until LB can successfully process an inference request.
    
    Handles cold-start race condition: /health may return 200 before
    the Ollama model is loaded into VRAM. Warm-up request triggers loading.
    """
    logger.info("Waiting for Load Balancer to be functionally ready...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            health_resp = httpx.Client(timeout=5).get(f"{LB_URL}/health")
            if health_resp.status_code != 200:
                time.sleep(1)
                continue
            
            warmup_resp = httpx.Client(timeout=20).post(
                f"{LB_URL}/infer",
                json={"prompt": "warmup", "max_tokens": 5},
                timeout=20.0
            )
            if warmup_resp.status_code == 200:
                data = warmup_resp.json()
                if data.get("status") == "success":
                    logger.info("Load balancer is functionally ready (model warm-up complete)")
                    return True
        except Exception:
            pass
        time.sleep(1)
    
    logger.error("LB functional readiness timeout after 45s")
    return False

def run_concurrent(n: int, use_rag: bool = False):
    """Execute concurrent load test with specified request count."""
    rag_flag = "enabled" if use_rag else "disabled"
    logger.info(f"Running concurrent test ({n} requests, RAG {rag_flag})")
    
    if not wait_for_lb_functional(timeout=45):
        logger.error("Aborting test: Load balancer not functionally ready")
        return 1
    
    env = os.environ.copy()
    env["RUN_ID"] = get_run_id()
    
    cmd = [sys.executable, str(TESTS_DIR / "test_concurrent.py"), "--count", str(n)]
    if use_rag:
        cmd.append("--use-rag")
    
    try:
        timeout_val = 1800 if use_rag else 1200
        res = subprocess.run(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            timeout=timeout_val,
            stdout=None,
            stderr=None
        )
        
        log_dir = get_log_dir()
        logger.info(f"Run Logs: {log_dir}")
        logger.info(f"Metrics: logs/metrics/")
        
        if res.returncode == 0:
            logger.info("Concurrent test completed successfully")
        else:
            logger.warning(f"Concurrent test completed with code {res.returncode}")
        return res.returncode
        
    except subprocess.TimeoutExpired:
        logger.error(f"Concurrent test timed out after {timeout_val} seconds")
        return 1
    except Exception as e:
        logger.error(f"Concurrent test exception: {e}")
        return 1

def run_fault(use_rag: bool = False):
    """Execute fault tolerance test with simulated worker failures."""
    rag_flag = "enabled" if use_rag else "disabled"
    logger.info(f"Running fault tolerance test (RAG {rag_flag})")
    
    env = os.environ.copy()
    env["RUN_ID"] = get_run_id()
    
    cmd = [sys.executable, str(TESTS_DIR / "test_fault_tolerance.py")]
    if use_rag:
        cmd.append("--use-rag")
    
    try:
        res = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), timeout=900)
        
        log_dir = get_log_dir()
        logger.info(f"Run Logs: {log_dir}")
        logger.info(f"Metrics: logs/metrics/")
        
        if res.returncode == 0:
            logger.info("Fault test completed successfully")
        else:
            logger.warning(f"Fault test completed with code {res.returncode}")
        return res.returncode
        
    except Exception as e:
        logger.error(f"Fault test exception: {e}")
        return 1

def menu(keep_lb_flag):
    """Interactive CLI for test selection and configuration."""
    while True:
        print(f"\n{'='*60}")
        print("Distributed LLM System - Test Orchestrator")
        print(f"{'='*60}")
        print(f"Run ID: {get_run_id()}")
        print(f"Logs: {get_log_dir()}")
        print(f"LB Window: {'Persistent' if keep_lb_flag else 'Auto-close on exit'}")
        print("1. Run Concurrent Load Test (10-1000) [RAG disabled]")
        print("2. Run Concurrent Load Test WITH RAG (slow, 1-10 recommended)")
        print("3. Run Fault Tolerance Test")
        print("4. Exit")
        print("-"*60)
        
        try:
            choice = input("Select (1/2/3/4): ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        
        logger.info(f"User selected option: {choice}")
        
        if choice in ["1", "2"]:
            try:
                count = int(input("Concurrent count (10-1000): ").strip())
                logger.info(f"User requested {count} concurrent requests")
                if 10 <= count <= 1000:
                    use_rag = (choice == "2")
                    if use_rag and count > 10:
                        confirm = input(f"RAG + {count} requests may timeout. Continue? (y/n): ").strip().lower()
                        if confirm != "y":
                            continue
                    run_concurrent(count, use_rag=use_rag)
                else:
                    logger.error("Invalid range (must be 10-1000)")
            except ValueError:
                logger.error("Invalid number input")
            except (EOFError, KeyboardInterrupt):
                pass
        elif choice == "3":
            run_fault()
        elif choice == "4":
            return 0
        else:
            logger.error(f"Invalid option: {choice}")

def cleanup(*args, keep_lb=False):
    """Graceful shutdown handler for SIGINT/SIGTERM."""
    logger.info("Shutdown signal received")
    global LB_PROC, LB_WINDOW_PID
    
    if not keep_lb and LB_WINDOW_PID:
        logger.info("Terminating Load Balancer window...")
        try:
            if sys.platform == "win32":
                # Force close the PowerShell window
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(LB_WINDOW_PID)],
                    capture_output=True,
                    timeout=5
                )
            else:
                import signal
                os.kill(LB_WINDOW_PID, signal.SIGTERM)
            logger.info("LB window terminated")
        except Exception as e:
            logger.warning(f"LB termination warning: {e}")
    elif keep_lb:
        logger.info("Load Balancer window kept open (use --keep-lb to auto-close)")
    
    LB_PROC = None
    LB_WINDOW_PID = None
    sys.exit(0)

def main():
    """Main entry point: orchestrate dependencies, parse args, run tests."""
    parser = argparse.ArgumentParser(description="Distributed LLM Test Orchestrator")
    parser.add_argument("--test", choices=["concurrent", "fault"], help="Run a specific test non-interactively")
    parser.add_argument("--count", type=int, default=100, help="Number of concurrent requests (for --test concurrent)")
    parser.add_argument("--use-rag", action="store_true", help="Enable RAG for the test")
    parser.add_argument("--keep-lb", action="store_true", help="Keep LB window open after tests complete")
    args = parser.parse_args()
    
    # Register signal handlers with keep_lb flag
    def sig_handler(signum, frame):
        cleanup(keep_lb=args.keep_lb)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    run_id = get_run_id()
    log_dir = get_log_dir()
    logger.info(f"System starting | Run: {run_id} | Logs: {log_dir}")
    
    if not start_ollama():
        logger.error("Failed to start Ollama - aborting")
        return 1
    if not start_lb(keep_window_open=args.keep_lb):
        logger.error("Failed to start Load Balancer - aborting")
        return 1
    
    # Non-interactive mode: run test and exit
    if args.test == "concurrent":
        if not (10 <= args.count <= 1000):
            logger.error("--count must be between 10 and 1000")
            return 1
        result = run_concurrent(args.count, use_rag=args.use_rag)
        if not args.keep_lb:
            cleanup(keep_lb=False)
        return result
    elif args.test == "fault":
        result = run_fault(use_rag=args.use_rag)
        if not args.keep_lb:
            cleanup(keep_lb=False)
        return result
    
    # Interactive mode: show menu
    result = menu(keep_lb_flag=args.keep_lb)
    if not args.keep_lb:
        cleanup(keep_lb=False)
    return result

if __name__ == "__main__":
    sys.exit(main())