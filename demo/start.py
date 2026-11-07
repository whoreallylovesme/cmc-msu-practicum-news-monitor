"""Start the Docker demo, run public-API E2E checks and leave its UI running."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PREFIX = "hn-monitor-demo"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["1k", "10k"], default="1k", help="Prelabelled snapshot size")
    parser.add_argument("--with-teacher", action="store_true", help="Also start the optional training profile")
    args = parser.parse_args()
    project = f"{PROJECT_PREFIX}-{args.dataset}"
    environment = {**os.environ, "MONITOR_DATASET_PATH": f"demo/hn_{args.dataset}.json"}
    compose = ["docker", "compose", "-p", project]
    if args.with_teacher:
        compose.extend(["--profile", "training"])
    try:
        resolved = subprocess.run(
            [*compose, "config", "--format", "json"], cwd=ROOT, env=environment,
            check=True, capture_output=True, text=True,
        )
        services = json.loads(resolved.stdout)["services"]
        api_port = int(services["web_master"]["ports"][0]["published"])
        ui_port = int(services["visualization"]["ports"][0]["published"])
        teacher_port = int(services["teacher"]["ports"][0]["published"]) if args.with_teacher else None
        for name in ("MONITOR_ADMIN_USER", "MONITOR_ADMIN_PASSWORD", "MONITOR_VIEWER_USER", "MONITOR_VIEWER_PASSWORD"):
            environment[name] = services["web_master"]["environment"][name]
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
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        print(f"Demo command failed with status {error.returncode}. Containers were left available for inspection.", file=sys.stderr)
        return error.returncode or 1
    except KeyboardInterrupt:
        print("Demo interrupted. Containers were left running.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
