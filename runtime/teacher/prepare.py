"""Prefetch and verify this teacher's pinned assets before offline operation."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import SETTINGS, prepare_assets
from lib.cli import configure_logging, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="Override TEACHER_ENCODER_CACHE_DIR/config cache_dir")
    args = parser.parse_args()
    paths = prepare_assets(SETTINGS, args.cache_dir.resolve() if args.cache_dir else None)
    write_json({"revision": SETTINGS["revision"], "assets": {name: str(path) for name, path in paths.items()}})


if __name__ == "__main__":
    configure_logging()
    main()
