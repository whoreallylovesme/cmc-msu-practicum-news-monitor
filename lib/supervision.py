"""Pinned human target-polarity supervision, separate from collected HN data."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTION_VERSION = "human-target-span-v1"


def canonical_target_text(row, mark_target=True):
    """Mark only a supplied annotation span; never infer one from the text."""
    text = row["text"]
    if not isinstance(text, str):
        raise ValueError("Supervision text must be a string")
    if not mark_target or row.get("target_start") is None:
        return text
    start, end = row["target_start"], row.get("target_end")
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
            or text[start:end] != row["entity"]):
        raise ValueError("Annotated target span does not match its raw text")
    return text[:start] + "[TARGET]" + text[start:end] + "[/TARGET]" + text[end:]


def _load_news(config, mode="from_scratch", version=1, seed=42):
    """Return model inputs, partial six-slot targets and auditable sample metadata.

    Gold polarity controls the marginal training loss, not the model input.
    Incremental replay uses seeded ID hashes and never looks at label difficulty.
    """
    spec = config.get("auxiliary_supervision") or {}
    if not spec.get("path"):
        return [], [], {"enabled": False, "items": 0}
    if mode not in {"from_scratch", "incremental"}:
        raise ValueError("Auxiliary mode must be from_scratch or incremental")
    source = Path(spec["path"]).expanduser()
    source = source.resolve() if source.is_absolute() else (ROOT / source).resolve()
    expected = str(spec.get("sha256") or "").lower()
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        raise ValueError("Auxiliary supervision requires a pinned SHA-256")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise ValueError("Auxiliary supervision SHA-256 mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Auxiliary supervision needs nonempty, unique training examples")
    for row in rows:
        if (row.get("split") != "train" or row.get("gold_relevant") is not True
                or row.get("gold_polarity") not in {"negative", "neutral", "positive"}
                or not isinstance(row.get("entity"), str) or not row["entity"].strip()
                or not isinstance(row.get("id"), str) or not row["id"]
                or not isinstance(row.get("group_id"), str) or not row["group_id"]
                or row.get("target_start") is None):
            raise ValueError("Auxiliary rows must be human-labelled, relevant training targets with a known polarity")
        canonical_target_text(row)
    available = len(rows)
    if mode == "incremental":
        limit = spec.get("incremental_replay_items", 256)
        if type(limit) is not int or limit < 0:
            raise ValueError("Auxiliary incremental_replay_items must be nonnegative")
        rows = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{version}:{row['id']}".encode()).digest())[:limit]
    items, targets = [], []
    for row in rows:
        items.append({
            "id": "aux:" + row["id"], "story_id": "aux:" + row["group_id"],
            "entity": row["entity"], "text": canonical_target_text(row),
            "parent_text": row.get("parent_text") or "", "story_title": row.get("story_title") or "",
            "parent_id": None, "polarity_only": True,
        })
        if config.get("student_encoder", {}).get("training_balance") is not None:
            items[-1]["training_source"] = row.get("dataset") or "news"
        target = [0.0] * 6
        target[{"negative": 1, "neutral": 2, "positive": 3}[row["gold_polarity"]]] = 1.0
        targets.append(target)
    return items, targets, {
        "enabled": True, "path": str(source), "sha256": digest,
        "projection_version": PROJECTION_VERSION, "mode": mode, "version": version, "seed": seed,
        "available_items": available, "items": len(items),
        "selected_ids": [item["id"] for item in items], "source_ids": [row["id"] for row in rows],
        "target_semantics": "Marginal negative A+B, neutral C, positive D+E; no intensity or off-topic labels",
    }


def load_auxiliary(config, mode="from_scratch", version=1, seed=42):
    """Keep News-only revisions unchanged; optionally append pinned CSV sources."""
    spec = config.get("auxiliary_supervision") or {}
    if not spec.get("additional_sources"):
        return _load_news(config, mode, version, seed)
    if mode not in {"from_scratch", "incremental"}:
        raise ValueError("Auxiliary mode must be from_scratch or incremental")
    from lib.training_sources import FORMAT_VERSION, identity, load_sources

    items, targets, news = _load_news(config, "from_scratch", version, seed)
    extra_items, extra_targets, sources = load_sources(config)
    items, targets = items + extra_items, targets + extra_targets
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Combined auxiliary sources repeat training IDs")
    available = len(items)
    revision = identity({"format_version": FORMAT_VERSION,
                         "news": {"sha256": news.get("sha256"), "projection_version": news.get("projection_version")},
                         "sources": [{key: source[key] for key in ("name", "format", "format_version", "source_sha256", "selection_sha256")} for source in sources]})
    if mode == "incremental":
        limit = spec.get("incremental_replay_items", 256)
        if type(limit) is not int or limit < 0:
            raise ValueError("Auxiliary incremental_replay_items must be nonnegative")
        indices = sorted(range(len(items)), key=lambda index: hashlib.sha256(f"{seed}:{version}:{items[index]['id'].removeprefix('aux:')}".encode()).digest())[:limit]
        items, targets = [items[index] for index in indices], [targets[index] for index in indices]
    selected = [item["id"].removeprefix("aux:") for item in items]
    news_ids = set(news.get("source_ids", []))
    for source in sources:
        groups = dict(zip(source["source_ids"], source["group_ids"]))
        source["available_items"] = source["items"]
        source["source_ids"] = [source_id for source_id in selected if source_id in groups]
        source["group_ids"] = [groups[source_id] for source_id in source["source_ids"]]
        source["items"] = len(source["source_ids"])
    semantics = "Human partial groups A+B, C, D+E, F; no invented five-level intensity labels"
    if any(source["format"] == "semeval_topic_five_point_zip" for source in sources):
        semantics += "; SemEval retains its original human five-point A-E topic polarity"
    if any(source["format"] == "canonical_supervision_jsonl" for source in sources):
        semantics += "; canonical sources retain their provided label distributions and annotation origins"
    return items, targets, {
        "enabled": True, "sha256": revision, "projection_version": FORMAT_VERSION,
        "mode": mode, "version": version, "seed": seed, "available_items": available, "items": len(items),
        "selected_ids": [item["id"] for item in items], "source_ids": [item["id"].removeprefix("aux:") for item in items],
        "group_ids": [item["story_id"].removeprefix("aux:") for item in items],
        "news": {"sha256": news.get("sha256"), "projection_version": news.get("projection_version"),
                 "available_items": news["items"], "items": sum(source_id in news_ids for source_id in selected)},
        "sources": sources,
        "target_semantics": semantics,
    }
