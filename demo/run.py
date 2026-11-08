"""Run the five HTTP services and an optional independent teacher."""
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.cli import configure_logging, load_cli_config


LOGGER = logging.getLogger(__name__)
SERVICES = ("collector", "storage", "ml", "web_master", "visualization")


def stop_on_term(_number, _frame):
    raise KeyboardInterrupt


def main():
    _config_path, settings = load_cli_config(ROOT / "demo/1k.yaml")
    dataset = settings.get("dataset", "1k")
    if dataset not in {"1k", "10k"}:
        raise ValueError("dataset must be 1k or 10k")
    with_teacher = settings.get("with_teacher", True)
    native = settings.get("native", {})
    base_port = int(native.get("base_port", 8100))
    configured_data = native.get("data_dir")
    data_dir = Path(configured_data) if configured_data else ROOT / ".local-data" / f"demo-{dataset}"
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    signal.signal(signal.SIGTERM, stop_on_term)
    services = SERVICES + (("teacher",) if with_teacher else ())
    if not 1024 <= base_port <= 65536 - len(services):
        raise ValueError("native.base_port must leave a valid port for every service")
    ports = {service: base_port + index for index, service in enumerate(services)}
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
    common.setdefault("MONITOR_DATASET_PATH", str(ROOT / "demo" / f"hn_{dataset}.json"))
    common.setdefault("ML_ENCODER_CACHE_DIR", str(data_dir.resolve() / "ml" / "encoder-weights"))
    if with_teacher:
        common["TEACHER_SERVICE_URL"] = urls["teacher"]
        common["TEACHER_URL"] = urls["teacher"]
        common.setdefault("TEACHER_MODEL_CACHE_DIR", str(data_dir.resolve() / "teacher" / "weights"))
        common.setdefault("TEACHER_LOG_DIR", str(data_dir.resolve() / "teacher" / "logs"))
    common.setdefault("OMP_NUM_THREADS", "2")
    common.setdefault("OPENBLAS_NUM_THREADS", "2")
    processes = {}
    teacher_exit_reported = False
    try:
        for service, port in ports.items():
            service_data = data_dir.resolve() / service
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
