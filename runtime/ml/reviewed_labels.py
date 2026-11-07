"""Apply assistant-reviewed labels to an already selected training subset."""

import hashlib
import json
from pathlib import Path

from lib.sentiment import CHOICES


DEFAULT_WEIGHT = 3
LABEL_SMOOTHING = 0.05
FINGERPRINT_FIELDS = ("text", "parent_text", "story_title", "story_id", "parent_id")


def fingerprint(item):
    """Hash raw label context, independently of Storage's content hash."""
    fields = {key: item.get(key) for key in FINGERPRINT_FIELDS}
    raw = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validated_rows(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Reviewed labels must contain an items list")
    if payload.get("training_only") is not True:
        raise ValueError("Reviewed labels must explicitly be training-only")
    reviewed = {}
    for row in payload["items"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Every reviewed label must have a comment id")
        label = row.get("label")
        digest = row.get("fingerprint")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(CHOICES):
            raise ValueError(f"Invalid reviewed class for comment {row['id']}")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid reviewed fingerprint for comment {row['id']}")
        if row["id"] in reviewed:
            raise ValueError(f"Duplicate reviewed label for comment {row['id']}")
        reviewed[row["id"]] = row
    return reviewed


def _distribution(label):
    values = [LABEL_SMOOTHING / len(CHOICES)] * len(CHOICES)
    values[label] += 1.0 - LABEL_SMOOTHING
    return values


def supervision_hash(row):
    """Detect changed targets independently of unchanged comment text."""
    raw = json.dumps(row["probabilities"], separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_reviewed_snapshot(path, entity):
    """Load a training reference without claiming neural-teacher provenance."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    reviewed = _validated_rows(payload)
    subject = payload.get("entity")
    if not isinstance(subject, str) or subject.strip().casefold() != str(entity).strip().casefold():
        raise ValueError(f"Reviewed labels are for {subject!r}, not {entity!r}")
    if not reviewed:
        raise ValueError("Reviewed labels contain no training examples")
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "entity": subject,
        "kind": "reviewed",
        "namespace": "reviewed-" + digest,
        "teacher": {
            "kind": "assistant_reviewed",
            "annotator": payload.get("annotator"),
            "sourcefile_sha": digest,
            "training_only": True,
            "limitation": payload.get("limitation"),
        },
        "labels": {
            comment_id: {"fingerprint": row["fingerprint"], "probabilities": _distribution(row["label"])}
            for comment_id, row in reviewed.items()
        },
    }


def apply_reviewed(items, targets, entity, path, weight=DEFAULT_WEIGHT):
    """Replace matching teacher targets and give reviewed rows ``weight`` copies.

    The caller must split train/holdout first and supply only training rows.
    Paths are used as supplied; resolving a configured relative path belongs to
    the caller. These labels are assistant review, not human evaluation gold.
    """
    items, targets = list(items), list(targets)
    if len(items) != len(targets):
        raise ValueError("Training items and targets must have equal lengths")
    if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
        raise ValueError("Reviewed label weight must be a positive integer")
    metadata = {
        "reviewed_items": 0,
        "weight": weight,
        "copies_added": 0,
        "sourcefile_sha": None,
        "status": "disabled",
    }
    if not path:
        return items, targets, metadata
    source = Path(path)
    if not source.exists():
        metadata["status"] = "missing"
        return items, targets, metadata

    raw = source.read_bytes()
    metadata["sourcefile_sha"] = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Reviewed labels must contain an items list")
    source_entity = payload.get("entity")
    metadata["source_entity"] = source_entity
    if not isinstance(source_entity, str) or source_entity.strip().casefold() != str(entity).strip().casefold():
        metadata.update(status="foreign_entity", skipped_entity=len(payload["items"]))
        return items, targets, metadata
    reviewed = _validated_rows(payload)

    expanded_items, expanded_targets = [], []
    for item, target in zip(items, targets):
        row = reviewed.get(str(item["id"]))
        copies = 1
        if row is not None:
            if fingerprint(item) != row["fingerprint"]:
                raise ValueError(f"Stale reviewed label for comment {item['id']}: raw context changed")
            target = _distribution(row["label"])
            copies = weight
            metadata["reviewed_items"] += 1
            metadata["copies_added"] += weight - 1
        expanded_items.extend([item] * copies)
        expanded_targets.extend([target] * copies)
    metadata["status"] = "applied"
    return expanded_items, expanded_targets, metadata
