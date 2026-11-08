"""Project checksum-pinned original human training sources into model supervision."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import zipfile
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
    target_mode = selected.get("target_mode", "distribution")
    if target_mode not in ("distribution", "class", "mixed"):
        raise ValueError("Weather target_mode must be distribution, class or mixed")
    values = [float(row[f"s{i}"]) for i in range(1, 6)]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values) or sum(values) <= 0:
        raise ValueError("Weather shares must be finite, nonnegative and nonempty")
    normalized = [value / sum(values) for value in values]
    weight = 1.0 - normalized[0]
    known = sum(normalized[1:])
    if not known:
        return None
    probabilities = [value / known for value in normalized[1:]]
    if target_mode in {"class", "mixed"}:
        winner = max(range(5), key=lambda index: values[index])
        if winner == 0 or values.count(values[winner]) != 1:
            return None
        mix = 1.0 if target_mode == "class" else selected.get("class_mix", 0.5)
        if isinstance(mix, bool) or not isinstance(mix, (int, float)) or not math.isfinite(mix) or not 0 <= mix <= 1:
            raise ValueError("Weather class_mix must be a finite number from zero to one")
        probabilities = [(1 - mix) * value + mix * (index == winner - 1)
                         for index, value in enumerate(probabilities)]
    return entity, row["tweet"], probabilities, weight


def _airline(row, _selected):
    if row["airline"] not in AIRLINES or row["airline_sentiment"] not in {"negative", "neutral", "positive"}:
        raise ValueError("Selected airline row has an unsupported source target or polarity")
    probabilities = [0.0] * 4
    probabilities[{"negative": 0, "neutral": 1, "positive": 2}[row["airline_sentiment"]]] = 1.0
    return AIRLINES[row["airline"]], row["text"], probabilities, 1.0


def _semeval_zip(source, selected, requested):
    """Read exactly one pinned original TRAIN member; never extract the archive."""
    member = selected.get("member")
    if not isinstance(member, str):
        raise ValueError("SemEval requires an explicit original TRAIN member")
    member_path = PurePosixPath(member)
    basename = re.fullmatch(r"twitter-(\d{4})train-CE\.txt", member_path.name)
    if (member_path.is_absolute() or ".." in member_path.parts or str(member_path) != member
            or any(part.casefold() in {"dev", "devtest", "test"} for part in member_path.parts)
            or not basename):
        raise ValueError("SemEval may read only its original TRAIN member, never DEV or TEST")
    expected_sha = selected.get("member_sha256")
    expected_size = selected.get("member_size_bytes")
    if (not isinstance(expected_sha, str) or len(expected_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha)
            or type(expected_size) is not int or not 0 < expected_size <= 64 * 1024 ** 2):
        raise ValueError("SemEval TRAIN member requires pinned SHA-256 and byte size")
    with zipfile.ZipFile(source) as archive:
        info = archive.getinfo(member)
        if info.file_size != expected_size:
            raise ValueError("SemEval TRAIN member byte size mismatch")
        with archive.open(info) as handle:
            raw = handle.read(expected_size + 1)
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("SemEval TRAIN member SHA-256 or byte size mismatch")
    found, seen = {}, set()
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t", 3)
        if len(fields) != 4:
            raise ValueError("SemEval TRAIN needs tweet ID, topic, five-point label and raw text")
        tweet, topic, label, text = fields
        if not tweet.isdecimal() or not topic.strip() or label not in {"-2", "-1", "0", "1", "2"}:
            raise ValueError("SemEval requires original human topic labels -2,-1,0,1,2")
        source_id = ("semeval" + basename.group(1) + "C:" + tweet + ":"
                     + hashlib.sha256(topic.encode()).hexdigest()[:16])
        if source_id not in requested:
            continue
        if source_id in seen:
            raise ValueError("SemEval TRAIN repeats a selected tweet/topic ID")
        seen.add(source_id)
        probabilities = [0.0] * 6
        probabilities[int(label) + 2] = 1.0
        found[source_id] = (topic, text, probabilities, 1.0)
    return found, seen


def _canonical_source(spec, config, cache):
    source, asset = _cached(spec["asset"], cache, 64 * 1024 ** 2)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Canonical supervision needs nonempty, unique TRAIN examples")
    items, targets = [], []
    for row in rows:
        if (row.get("split") != "train" or type(row.get("partial_labels")) is not bool
                or any(not isinstance(row.get(key), str) or not row[key].strip()
                       for key in ("id", "group_id", "entity", "text"))):
            raise ValueError("Canonical supervision requires explicit TRAIN targets, IDs and label semantics")
        values = row["label_probabilities"]
        if not isinstance(values, list) or len(values) != 6:
            raise ValueError("Canonical supervision requires six source probabilities")
        values = [float(value) for value in values]
        weight = float(row.get("sample_weight", 1))
        if (any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0
                or not math.isfinite(weight) or not 0 < weight <= 1):
            raise ValueError("Canonical source probabilities and confidence must be finite and valid")
        parent, title = row.get("parent_text") or "", row.get("story_title") or ""
        if not isinstance(parent, str) or not isinstance(title, str):
            raise ValueError("Canonical supervision context must be original text")
        items.append({"id": "aux:" + row["id"], "story_id": "aux:" + row["group_id"],
                      "entity": row["entity"], "text": row["text"], "parent_text": parent,
                      "story_title": title, "parent_id": row.get("parent_id"),
                      "partial_labels": row["partial_labels"], "sample_weight": weight})
        if config.get("student_encoder", {}).get("training_balance") is not None:
            items[-1]["training_source"] = spec["name"]
        targets.append(values)
    return items, targets, {
        "name": spec["name"], "format": spec["format"], "format_version": "canonical-supervision-jsonl-v1",
        "source_sha256": asset.sha256, "source_size_bytes": asset.size_bytes,
        "selection_sha256": None, "selection_size_bytes": 0,
        "source_ids": [row["id"] for row in rows], "group_ids": [row["group_id"] for row in rows],
        "skipped_uninformative_ids": [], "items": len(rows),
        "label_origin": spec.get("label_origin", "provided"),
        "target_semantics": "Original source distributions and partial-label flags; no inferred spans or relabelling",
    }


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
        if format_name == "canonical_supervision_jsonl":
            extra_items, extra_targets, metadata = _canonical_source(spec, config, cache)
            items.extend(extra_items)
            targets.extend(extra_targets)
            sources.append(metadata)
            continue
        if format_name not in {"crowdflower_weather_csv", "airline_csv", "semeval_topic_five_point_zip"}:
            raise ValueError("Unsupported human training source format")
        selection_path, selection_asset = _cached(spec["selection"], cache, 16 * 1024 ** 2)
        raw_spec = asset_spec(spec["asset"], max_bytes=64 * 1024 ** 2, base_dir=ROOT)
        selected = _selection(selection_path, name, format_name, raw_spec.sha256)
        source_path, source_asset = _cached(spec["asset"], cache, 64 * 1024 ** 2)
        requested = {row["id"]: row["group_id"] for row in selected["items"]}
        if format_name == "semeval_topic_five_point_zip":
            found, seen = _semeval_zip(source_path, selected, requested)
        else:
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
            raise ValueError("Training source is missing selected source IDs")
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
            full_five = format_name == "semeval_topic_five_point_zip"
            items.append({"id": "aux:" + source_id, "story_id": "aux:" + group_id,
                          "entity": entity, "text": text, "parent_text": "", "story_title": "",
                          "parent_id": None, "partial_labels": not full_five, "sample_weight": weight})
            if config.get("student_encoder", {}).get("training_balance") is not None:
                items[-1]["training_source"] = name
            targets.append(probabilities if full_five else
                           [0, probabilities[0], probabilities[1], probabilities[2], 0, probabilities[3]])
            kept.append(source_id)
        metadata = {"name": name, "format": format_name, "format_version": FORMAT_VERSION,
                    "source_sha256": source_asset.sha256, "source_size_bytes": source_asset.size_bytes,
                    "selection_sha256": selection_asset.sha256, "selection_size_bytes": selection_asset.size_bytes,
                    "source_ids": kept, "group_ids": [requested[source_id] for source_id in kept],
                    "skipped_uninformative_ids": skipped, "items": len(kept)}
        if format_name == "crowdflower_weather_csv" and selected.get("target_mode") in {"class", "mixed"}:
            mode = selected["target_mode"]
            metadata.update(format_version=f"selected-human-csv-{mode}-v1", target_mode=mode)
            if mode == "mixed":
                metadata["class_mix"] = selected.get("class_mix", 0.5)
        if format_name == "semeval_topic_five_point_zip":
            metadata.update(format_version="selected-semeval-topic-five-v1",
                            member=selected["member"], member_sha256=selected["member_sha256"],
                            member_size_bytes=selected["member_size_bytes"], known_related=True,
                            target_semantics="Original human five-point topic polarity A-E; no F examples")
        sources.append(metadata)
    return items, targets, sources
