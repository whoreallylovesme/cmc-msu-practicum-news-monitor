"""Prefetch the optional semantic encoder's pinned inference assets."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime/ml")]

from lib.config import load_config, load_monitoring, load_project_config
from lib.mentions import MentionMatcher
import encoder
import nli_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get(
        "ML_ENCODER_CACHE_DIR", ROOT / ".local-data/ml/encoder-weights",
    )))
    parser.add_argument("--features", action="store_true", help="Also prepare label-free features of the offline snapshot")
    args = parser.parse_args()
    os.environ["ML_ENCODER_CACHE_DIR"] = str(args.cache_dir.resolve())
    config = load_config(ROOT / "runtime/ml/config.yaml", env_var="ML_CONFIG")
    monitoring = load_monitoring()
    config.update(monitoring=monitoring, semantic_matching=monitoring["matching"])
    bundle = encoder.prepare(config)
    if bundle.get("output_mode") == "nli" or bundle.get("backend") == "nli":
        paths = nli_encoder.ensure_assets(bundle, args.cache_dir.resolve())
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        if args.features:
            dataset = Path(load_project_config()["path_to_offline_demo_data"])
            items = [item for story in json.loads(dataset.read_text(encoding="utf-8")) for item in story["comments"]]
            matcher = MentionMatcher(monitoring["query"], monitoring["aliases"], monitoring["matching"])
            items = [item for item in items if matcher.match(item) is not None]
            selected = config["student_encoder"].get("feature_cache")
            destination = ROOT / "runtime/ml" / selected if selected else args.cache_dir.resolve() / "nli_features.json"
            nli_encoder.generate_feature_cache(config, items, destination, monitoring["query"])
            print(f"Prepared features: {destination}")
    else:
        print("Encoder assets are available")


if __name__ == "__main__":
    main()
