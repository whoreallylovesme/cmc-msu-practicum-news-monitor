"""Logging and machine-readable output for command-line entry points."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path


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


def load_cli_config(default_path, argv=None):
    """CLI entry points accept a YAML path; operation settings live in that file."""
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_path, help="Path to the YAML configuration")
    path = parser.parse_args(argv).config.expanduser().resolve()
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return path, config
