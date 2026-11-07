"""Start the Docker demo, run public-API E2E checks and leave its UI running."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.cli import configure_logging, load_cli_config


LOGGER = logging.getLogger(__name__)
PROJECT_PREFIX = "hn-monitor-demo"


def main():
    config_path, settings = load_cli_config(ROOT / "demo/1k.yaml")
    dataset = settings.get("dataset", "1k")
    if dataset not in {"1k", "10k"}:
        raise ValueError("dataset must be 1k or 10k")
    with_teacher = settings.get("with_teacher", True)
    project = f"{PROJECT_PREFIX}-{dataset}"
    environment = {**os.environ, "MONITOR_DATASET_PATH": f"demo/hn_{dataset}.json"}
    for name, default in (("api", 8081), ("ui", 8080), ("teacher", 8082)):
        port = int(settings.get("ports", {}).get(name, default))
        if not 1 <= port <= 65535:
            raise ValueError(f"ports.{name} must be between 1 and 65535")
        environment[f"MONITOR_{name.upper()}_PORT"] = str(port)
    compose = ["docker", "compose", "-p", project]
    if with_teacher:
        compose.extend(["--profile", "training"])
    try:
        resolved = subprocess.run(
            [*compose, "config", "--format", "json"], cwd=ROOT, env=environment,
            check=True, capture_output=True, text=True,
        )
        services = json.loads(resolved.stdout)["services"]
        api_port = int(services["web_master"]["ports"][0]["published"])
        ui_port = int(services["visualization"]["ports"][0]["published"])
        teacher_port = int(services["teacher"]["ports"][0]["published"]) if with_teacher else None
        for name in ("MONITOR_ADMIN_USER", "MONITOR_ADMIN_PASSWORD", "MONITOR_VIEWER_USER", "MONITOR_VIEWER_PASSWORD"):
            environment[name] = services["web_master"]["environment"][name]
        environment["MONITOR_E2E_URL"] = f"http://127.0.0.1:{api_port}"
        subprocess.run([*compose, "up", "--build", "--wait"], cwd=ROOT, env=environment, check=True)
        LOGGER.info("UI: http://127.0.0.1:%s/", ui_port)
        LOGGER.info("Swagger API: http://127.0.0.1:%s/gateway-docs", ui_port)
        if teacher_port is not None:
            LOGGER.info("Teacher Swagger: http://127.0.0.1:%s/docs", teacher_port)
        subprocess.run(
            [sys.executable, str(ROOT / "demo/e2e.py"), "--config", str(config_path)],
            cwd=ROOT, env=environment, check=True,
        )
        LOGGER.info("Demo is running. From the project root, stop it with: docker compose -p %s down", project)
        return 0
    except FileNotFoundError as error:
        LOGGER.error("Could not start the demo command: %s", error)
        return 127
    except subprocess.CalledProcessError as error:
        if error.stderr:
            LOGGER.error("%s", error.stderr.rstrip())
        LOGGER.error("Demo command failed with status %s. Containers were left available for inspection.", error.returncode)
        return error.returncode or 1
    except KeyboardInterrupt:
        LOGGER.info("Demo interrupted. Containers were left running.")
        return 130


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
