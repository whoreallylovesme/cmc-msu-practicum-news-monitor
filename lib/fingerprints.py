"""Canonical content and prepared-label context fingerprints."""

import hashlib
import json


COMMENT_CONTENT_FIELDS = ("story_id", "story_title", "parent_id", "parent_text", "text")
LABEL_CONTEXT_FIELDS = ("text", "parent_text", "story_title", "story_id", "parent_id")


def comment_content_hash(item):
    fields = [item.get(key) for key in COMMENT_CONTENT_FIELDS]
    raw = json.dumps(fields, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def label_context_fingerprint(item):
    fields = {key: item.get(key) for key in LABEL_CONTEXT_FIELDS}
    raw = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
