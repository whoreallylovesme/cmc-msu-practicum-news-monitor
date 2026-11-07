"""Collect comments, train when selected labels are available, then refresh scores."""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.client import call, monitoring_options, wait_job, wait_ready


def update(args):
    wait_ready(args.url, 30)
    entity, source, teacher = monitoring_options(
        args.url, entity=args.entity, source=args.source, teacher=args.teacher,
    )
    model_path = f"/api/models/{args.model_id}"
    model = call(args.url, model_path + "/summary")
    if model["entity"] != entity:
        raise RuntimeError("The model belongs to a different entity")
    if model.get("version") is None:
        raise RuntimeError("Train the model from scratch before starting daily updates")

    cursor = None
    received = 0
    for _ in range(args.pages):
        batch = call(args.url, "/api/import", body={
            "entity": entity,
            "source": source,
            "limit": args.batch_size,
            "cursor": cursor,
        })
        received += batch["received"]
        cursor = batch["next_cursor"]
        if not cursor:
            break

    job = call(args.url, model_path + "/train", body={
        "teacher": teacher, "limit": args.training_limit, "mode": "incremental",
    })
    result = wait_job(args.url, job["job_id"], args.timeout)
    scoring = call(args.url, model_path + "/score", body={
        "entity": entity, "limit": args.score_limit,
    })
    print(json.dumps({
        "at": datetime.now(timezone.utc).isoformat(),
        "received": received,
        "training": result["status"],
        "version": scoring["version"],
        "scored": scoring["stored"],
        "reason": result.get("reason"),
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--entity", help="Override the configured monitoring query")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source", help="Override the configured monitoring source")
    parser.add_argument("--teacher", choices=["dataset", "reviewed", "service", "live", "snapshot", "fixture"],
                        help="Label source; defaults to dataset labels for the Hacker News snapshot")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--training-limit", type=int, default=200)
    parser.add_argument("--score-limit", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--interval-hours", type=float, help="Repeat at this interval, e.g. 24")
    args = parser.parse_args()
    if args.pages < 1 or not 1 <= args.batch_size <= 100:
        parser.error("invalid batch or page limit")
    if args.interval_hours is not None and args.interval_hours <= 0:
        parser.error("interval-hours must be positive")

    while True:
        started = time.monotonic()
        try:
            update(args)
        except (RuntimeError, URLError) as error:
            message = error.read().decode(errors="replace") if isinstance(error, HTTPError) else str(error)
            print(message, file=sys.stderr, flush=True)
            if args.interval_hours is None:
                return 1
        if args.interval_hours is None:
            return 0
        time.sleep(max(0, args.interval_hours * 3600 - (time.monotonic() - started)))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
