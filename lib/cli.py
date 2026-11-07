"""Logging and machine-readable output for command-line entry points."""

import json
import logging
import os
import sys


def configure_logging():
    logging.basicConfig(
        level=os.environ.get("MONITOR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def write_json(value):
    """Keep command results on stdout and diagnostics on stderr."""
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    sys.stdout.flush()
