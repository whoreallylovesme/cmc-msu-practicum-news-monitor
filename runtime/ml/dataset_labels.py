"""Load prepared labels from the same real HN snapshot used by the demo."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from lib.sentiment import CHOICES
from lib.teacher_contracts import input_hash


FINGERPRINT_FIELDS = ("text", "parent_text", "story_title", "story_id", "parent_id")


def fingerprint(item):
    """Hash raw label context, independently of Storage's content hash."""
    fields = {key: item.get(key) for key in FINGERPRINT_FIELDS}
    raw = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def supervision_hash(row):
    """Detect changed targets independently of unchanged comment text."""
    raw = json.dumps({"probabilities": row["probabilities"], "teacher_input_hash": row.get("teacher_input_hash")},
                     sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_dataset_snapshot(path, entity, aliases=None):
    raw = Path(path).read_bytes()
    stories = json.loads(raw)
    if not isinstance(stories, list):
        raise ValueError("Prepared dataset must contain a list of discussions")
    labels = {}
    sources = Counter()
    for story in stories:
        for item in story.get("comments", []):
            comment_id = item["id"]
            label = item.get("label")
            subject = item.get("label_entity")
            if not isinstance(subject, str) or subject.casefold() != entity.casefold():
                raise ValueError(f"Prepared labels are for {subject!r}, not {entity!r}")
            if label not in CHOICES or not item.get("label_source"):
                raise ValueError(f"Missing prepared label or provenance for {comment_id}")
            digest = fingerprint(item)
            if item.get("label_fingerprint") != digest:
                raise ValueError(f"Stale prepared label for {comment_id}: raw context changed")
            if aliases is not None and item.get("label_input_hash") != input_hash(entity, item, aliases):
                raise ValueError(f"Stale prepared label for {comment_id}: target, aliases, ID or full context changed; "
                                 "relabel the dataset for the current target and aliases")
            if comment_id in labels:
                raise ValueError(f"Duplicate prepared label for {comment_id}")
            probabilities = item.get("label_probabilities")
            if (not isinstance(probabilities, list) or len(probabilities) != len(CHOICES)
                    or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
                    or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)
                    or CHOICES[max(range(len(CHOICES)), key=probabilities.__getitem__)] != label):
                raise ValueError(f"Invalid teacher distribution for {comment_id}; relabel the prepared dataset")
            labels[comment_id] = {"fingerprint": digest, "probabilities": probabilities,
                                  "teacher_input_hash": item.get("label_input_hash")}
            sources[item["label_source"]] += 1
    if not labels:
        raise ValueError("Prepared dataset contains no labeled comments")
    checksum = hashlib.sha256(raw).hexdigest()
    return {
        "entity": entity,
        "kind": "dataset",
        "namespace": "dataset-" + checksum,
        "teacher": {
            "kind": "prepared_dataset",
            "source_sha256": checksum,
            "items": len(labels),
            "label_sources": dict(sources),
            "limitation": "Prepared teacher distributions are silver labels, not independent human gold.",
        },
        "labels": labels,
    }
