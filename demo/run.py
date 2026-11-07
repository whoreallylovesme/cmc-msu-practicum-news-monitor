"""Run the five HTTP services and an optional independent teacher."""
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.cli import configure_logging


LOGGER = logging.getLogger(__name__)
SERVICES = ("collector", "storage", "ml", "web_master", "visualization")


def stop_on_term(_number, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dataset", choices=["1k", "10k"], help="Override the configured snapshot size")
    parser.add_argument("--with-teacher", action="store_true", help="Also start the independent labeling service")
    args = parser.parse_args()
    if args.data_dir is None:
        selected = args.dataset or Path(os.environ.get("MONITOR_DATASET_PATH", "hn_1k.json")).stem.removeprefix("hn_")
        args.data_dir = ROOT / ".local-data" / ("demo-" + selected if selected in {"1k", "10k"} else "native")
    if not 1024 <= args.base_port <= 65531:
        parser.error("base-port must be between 1024 and 65531")

    signal.signal(signal.SIGTERM, stop_on_term)
    services = SERVICES + (("teacher",) if args.with_teacher else ())
    if args.base_port + len(services) - 1 > 65535:
        parser.error("base-port leaves no port for the teacher")
    ports = {service: args.base_port + index for index, service in enumerate(services)}
    urls = {service: f"http://127.0.0.1:{port}" for service, port in ports.items()}
    common = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "COLLECTOR_URL": urls["collector"],
        "STORAGE_URL": urls["storage"],
        "ML_URL": urls["ml"],
        "VISUALIZATION_URL": urls["visualization"],
        "MASTER_URL": urls["web_master"],
        "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
    }
    if args.dataset is not None:
        common["MONITOR_DATASET_PATH"] = f"demo/hn_{args.dataset}.json"
    common.setdefault("ML_MODEL_CACHE_DIR", str(args.data_dir.resolve() / "ml" / "weights"))
    common.setdefault("ML_ENCODER_CACHE_DIR", str(args.data_dir.resolve() / "ml" / "encoder-weights"))
    if args.with_teacher:
        common["TEACHER_SERVICE_URL"] = urls["teacher"]
        common["TEACHER_URL"] = urls["teacher"]
        common.setdefault("TEACHER_ENCODER_CACHE_DIR", str(args.data_dir.resolve() / "teacher" / "encoder-weights"))
    common.setdefault("OMP_NUM_THREADS", "2")
    common.setdefault("OPENBLAS_NUM_THREADS", "2")
    processes = {}
    teacher_exit_reported = False
    try:
        for service, port in ports.items():
            service_data = args.data_dir.resolve() / service
            service_data.mkdir(parents=True, exist_ok=True)
            processes[service] = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
                cwd=ROOT / "runtime" / service,
                env={**common, "MONITOR_DATA_DIR": str(service_data)},
            )
            LOGGER.info("%s: %s", service, urls[service])

        while True:
            for service, process in processes.items():
                returncode = process.poll()
                if returncode is None:
                    continue
                if service == "teacher":
                    if not teacher_exit_reported:
                        LOGGER.warning("Optional teacher exited with status %s; the five main services will keep running.", returncode)
                        teacher_exit_reported = True
                    continue
                LOGGER.error("%s exited with status %s", service, returncode)
                return returncode or 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
