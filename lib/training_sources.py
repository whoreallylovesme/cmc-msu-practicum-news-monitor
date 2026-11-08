"""Project checksum-pinned CSV selections into partial human supervision."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from lib.model_assets import asset_spec, ensure_asset


ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = "selected-human-csv-v1"
AIRLINES = {"American": "American Airlines", "United": "United Airlines",
            "Southwest": "Southwest Airlines", "US Airways": "US Airways",
            "Virgin America": "Virgin America"}


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _cached(spec, cache, maximum):
    if type(spec.get("size_bytes")) is not int:
        raise ValueError("Training assets require a pinned size_bytes")
    checked = asset_spec(spec, max_bytes=maximum, base_dir=ROOT)
    return ensure_asset(checked, cache, suffix=Path(urlsplit(checked.path).path).suffix or ".bin", timeout=900), checked


def _selection(path, name, format_name, source_sha):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("Unknown training selection version")
    selected = document.get("sources", {}).get(name)
    if (not isinstance(selected, dict) or selected.get("format") != format_name
            or selected.get("asset", {}).get("sha256") != source_sha or selected.get("split") != "train"):
        raise ValueError("Selection must pin the matching training source and format")
    rows = selected.get("items")
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                   or not isinstance(row.get("group_id"), str) or not row["group_id"] for row in rows)
            or len({row["id"] for row in rows}) != len(rows)):
        raise ValueError("Training selection needs unique ordered IDs and group IDs")
    return selected


def _weather(row, selected):
    entity = selected.get("entity")
    if not isinstance(entity, str) or not entity.strip():
        raise ValueError("Weather supervision requires an explicit target entity")
    values = [float(row[f"s{i}"]) for i in range(1, 6)]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values) or sum(values) <= 0:
        raise ValueError("Weather shares must be finite, nonnegative and nonempty")
    normalized = [value / sum(values) for value in values]
    weight = 1.0 - normalized[0]
    known = sum(normalized[1:])
    if not known:
        return None
    probabilities = [value / known for value in normalized[1:]]
    return entity, row["tweet"], probabilities, weight


def _airline(row, _selected):
    if row["airline"] not in AIRLINES or row["airline_sentiment"] not in {"negative", "neutral", "positive"}:
        raise ValueError("Selected airline row has an unsupported source target or polarity")
    probabilities = [0.0] * 4
    probabilities[{"negative": 0, "neutral": 1, "positive": 2}[row["airline_sentiment"]]] = 1.0
    return AIRLINES[row["airline"]], row["text"], probabilities, 1.0


def load_sources(config):
    """Load the exact selected rows, without searching, relabelling or resampling."""
    specs = config.get("auxiliary_supervision", {}).get("additional_sources", [])
    if not isinstance(specs, list) or len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("Additional sources require a list of distinct source names")
    cache = Path(os.environ.get("MONITOR_DATA_DIR", config.get("data_dir", ".local-data"))).expanduser() / "training-sources"
    if not cache.is_absolute():
        cache = ROOT / cache
    items, targets, sources = [], [], []
    for spec in specs:
        name, format_name = spec["name"], spec["format"]
        if format_name not in {"crowdflower_weather_csv", "airline_csv"}:
            raise ValueError("Unsupported human training CSV format")
        selection_path, selection_asset = _cached(spec["selection"], cache, 4 * 1024 ** 2)
        raw_spec = asset_spec(spec["asset"], max_bytes=64 * 1024 ** 2, base_dir=ROOT)
        selected = _selection(selection_path, name, format_name, raw_spec.sha256)
        source_path, source_asset = _cached(spec["asset"], cache, 64 * 1024 ** 2)
        requested = {row["id"]: row["group_id"] for row in selected["items"]}
        found, seen = {}, set()
        with source_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "tweet", "s1", "s2", "s3", "s4", "s5"} if format_name == "crowdflower_weather_csv" else {"tweet_id", "airline", "airline_sentiment", "text"}
            if not required <= set(reader.fieldnames or []):
                raise ValueError("Training CSV does not contain its required original columns")
            for index, row in enumerate(reader):
                source_id = "partly-sunny:" + row["id"] if format_name == "crowdflower_weather_csv" else f"airline:{row['tweet_id']}:{index}"
                if source_id not in requested:
                    continue
                if source_id in seen:
                    raise ValueError("Training CSV repeats a selected source ID")
                seen.add(source_id)
                found[source_id] = (_weather if format_name == "crowdflower_weather_csv" else _airline)(row, selected)
        if seen != set(requested):
            raise ValueError("Training CSV is missing selected source IDs")
        kept, skipped = [], []
        for row in selected["items"]:
            source_id, group_id = row["id"], row["group_id"]
            result = found[source_id]
            if result is None:
                skipped.append(source_id)
                continue
            entity, text, probabilities, weight = result
            if not isinstance(text, str) or not text.strip() or not 0 < weight <= 1:
                raise ValueError("Training row needs raw text and a positive known-label mass")
            items.append({"id": "aux:" + source_id, "story_id": "aux:" + group_id,
                          "entity": entity, "text": text, "parent_text": "", "story_title": "",
                          "parent_id": None, "partial_labels": True, "sample_weight": weight})
            targets.append([0, probabilities[0], probabilities[1], probabilities[2], 0, probabilities[3]])
            kept.append(source_id)
        sources.append({"name": name, "format": format_name, "format_version": FORMAT_VERSION,
                        "source_sha256": source_asset.sha256, "source_size_bytes": source_asset.size_bytes,
                        "selection_sha256": selection_asset.sha256, "selection_size_bytes": selection_asset.size_bytes,
                        "source_ids": kept, "group_ids": [requested[source_id] for source_id in kept],
                        "skipped_uninformative_ids": skipped, "items": len(kept)})
    return items, targets, sources
