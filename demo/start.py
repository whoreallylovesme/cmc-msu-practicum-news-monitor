"""Start the Docker demo, run public-API E2E checks and leave its UI running."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PREFIX = "hn-monitor-demo"


def port_from_env(parser, name, default):
    try:
        port = int(os.environ.get(name, str(default)))
    except ValueError:
        parser.error(f"{name} must be an integer port")
    if not 1 <= port <= 65535:
        parser.error(f"{name} must be between 1 and 65535")
    return port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["1k", "10k"], default="1k", help="Prelabelled snapshot size")
    parser.add_argument("--with-teacher", action="store_true", help="Also start the optional training profile")
    args = parser.parse_args()
    api_port = port_from_env(parser, "MONITOR_API_PORT", 8081)
    ui_port = port_from_env(parser, "MONITOR_UI_PORT", 8080)
    teacher_port = port_from_env(parser, "MONITOR_TEACHER_PORT", 8082) if args.with_teacher else None
    project = f"{PROJECT_PREFIX}-{args.dataset}"
    environment = {**os.environ, "MONITOR_DATASET_PATH": f"demo/hn_{args.dataset}.json"}
    compose = ["docker", "compose", "-p", project]
    if args.with_teacher:
        compose.extend(["--profile", "training"])
    try:
        subprocess.run([*compose, "up", "--build", "--wait"], cwd=ROOT, env=environment, check=True)
        print(f"UI: http://127.0.0.1:{ui_port}/", flush=True)
        print(f"Swagger API: http://127.0.0.1:{ui_port}/gateway-docs", flush=True)
        if teacher_port is not None:
            print(f"Teacher Swagger: http://127.0.0.1:{teacher_port}/docs", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "demo/e2e.py"), "--url", f"http://127.0.0.1:{api_port}", "--dataset", args.dataset],
            cwd=ROOT, env=environment, check=True,
        )
        print(f"Demo is running. From the project root, stop it with: docker compose -p {project} down", flush=True)
        return 0
    except FileNotFoundError as error:
        print(f"Could not start the demo command: {error}", file=sys.stderr)
        return 127
    except subprocess.CalledProcessError as error:
        print(f"Demo command failed with status {error.returncode}. Containers were left available for inspection.", file=sys.stderr)
        return error.returncode or 1
    except KeyboardInterrupt:
        print("Demo interrupted. Containers were left running.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
