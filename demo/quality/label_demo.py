"""Полная разметка HN с возобновляемым кешем и публикацией после завершения."""

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

from lib.sentiment import CHOICES, SCORE_LEVELS
from lib.teacher_contracts import INPUT_HASH_VERSION, input_hash


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime/ml"))
from dataset_labels import FINGERPRINT_FIELDS, fingerprint


LOGGER = logging.getLogger(__name__)
ANNOTATIONS = {"fixture_label", "label", "label_entity", "label_source", "label_fingerprint",
               "label_probabilities", "label_input_hash"}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def source_context(stories):
    """Сравниваем UTF-8 исходных полей, включая весь текст без нормализации."""
    raw = [{**{key: value for key, value in story.items() if key != "comments"},
            "comments": [{key: value for key, value in item.items() if key not in ANNOTATIONS}
                         for item in story["comments"]]} for story in stories]
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def model_input(row):
    return {"id": str(row["id"]), "story_id": str(row["story_id"]), "text": row["text"],
            "story_title": row.get("story_title") or "", "parent_text": row.get("parent_text"),
            "parent_id": row.get("parent_id")}


def distribution(values):
    values = [float(value) for value in values]
    assert len(values) == len(CHOICES) and all(math.isfinite(value) and 0 <= value <= 1 for value in values)
    assert math.isclose(sum(values), 1.0, abs_tol=1e-6), "Учитель должен вернуть распределение A–F"
    return values


def annotate(stories, entity, aliases, predict, settings, cache):
    identity = {"version": "hn-soft-labels-v1", "input_hash_version": INPUT_HASH_VERSION,
                "classes": CHOICES, "teacher_protocol": settings["protocol"]}
    protocol_hash = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    parallel = (settings.get("llama") or {}).get("parallel", 1)
    assert type(parallel) is int and parallel > 0, "llama.parallel должен быть положительным целым"
    connection = sqlite3.connect(str(cache), timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""CREATE TABLE IF NOT EXISTS demo_labels (
        revision TEXT NOT NULL, protocol_hash TEXT NOT NULL, input_hash TEXT NOT NULL,
        probabilities TEXT NOT NULL, PRIMARY KEY (revision, protocol_hash, input_hash))""")
    result, cached, completed, started = copy.deepcopy(stories), 0, 0, time.perf_counter()
    total = sum(len(story["comments"]) for story in result)

    def assign(row, digest, values):
        nonlocal completed
        original_fingerprint = fingerprint(row)
        for key in ANNOTATIONS:
            row.pop(key, None)
        row.update(label=CHOICES[max(range(len(CHOICES)), key=values.__getitem__)],
                   label_probabilities=values, label_entity=entity, label_source=settings["revision"],
                   label_fingerprint=original_fingerprint, label_input_hash=digest)
        assert fingerprint(row) == original_fingerprint
        completed += 1
        if completed % 100 == 0:
            LOGGER.info("Размечено %d/%d; из кеша %d; прошло %.1fs", completed, total, cached, time.perf_counter() - started)

    try:
        missing = []
        for story in result:
            for row in story["comments"]:
                item = model_input(row)
                digest = input_hash(entity, item, aliases)
                saved = connection.execute("SELECT probabilities FROM demo_labels WHERE revision=? AND protocol_hash=? AND input_hash=?",
                                           (settings["revision"], protocol_hash, digest)).fetchone()
                if saved is None:
                    missing.append((row, item, digest))
                else:
                    cached += 1
                    assign(row, digest, distribution(json.loads(saved[0])))
        if missing:
            LOGGER.info("Новых запросов %d; одновременных запросов к одному учителю %d", len(missing), parallel)
            remaining = iter(missing)
            with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="hn-label") as executor:
                pending = {}

                def replenish():
                    while len(pending) < parallel:
                        entry = next(remaining, None)
                        if entry is None:
                            break
                        row, item, digest = entry
                        pending[executor.submit(predict, entity, item, aliases)] = (row, digest)

                try:
                    replenish()
                    while pending:
                        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in finished:
                            row, digest = pending.pop(future)
                            values = distribution(future.result())
                            # Only this coordinator accesses SQLite or source rows.
                            connection.execute("INSERT INTO demo_labels VALUES (?, ?, ?, ?)",
                                               (settings["revision"], protocol_hash, digest, json.dumps(values)))
                            connection.commit()
                            assign(row, digest, values)
                        replenish()
                except BaseException:
                    # At most parallel futures exist; wait only for those already
                    # running, never for an eagerly submitted 10000-item queue.
                    for future in pending:
                        future.cancel()
                    raise
    finally:
        connection.close()
    assert completed == total
    assert source_context(result) == source_context(stories), "Изменились исходные поля HN"
    return result, {"version": identity["version"], "sha256": protocol_hash}, cached


def metadata(previous, stories, raw, entity, aliases, settings, protocol):
    result = copy.deepcopy(previous)
    for key in ("matching", "mention_candidate_count", "prepared_labels", "subset_of", "subsets"):
        result.pop(key, None)
    rows = [row for story in stories for row in story["comments"]]
    dates = [datetime.fromisoformat(row["published_at"].replace("Z", "+00:00")).astimezone(timezone.utc) for row in rows]
    asset = settings["model"]
    source = {"kind": "local_file", "filename": Path(asset.path).name} if asset.is_local else {"kind": "https", "url": asset.path}
    counts = Counter(row["label"] for row in rows)
    result.update(query=entity, aliases=aliases, thread_count=len(stories), comment_count=len(rows),
                  sha256=sha256(raw), bytes=len(raw), min_comment_date=min(dates).isoformat(), max_comment_date=max(dates).isoformat(),
                  dataset_kind="public_discussion_snapshot_with_silver_labels")
    result["prepared_labels"] = {
        "entity": entity, "aliases": aliases, "comments": len(rows),
        "classes": {label: counts[label] for label in CHOICES}, "sources": {settings["revision"]: len(rows)},
        "class_meaning": dict(zip(CHOICES, ("strong_negative", "mild_negative", "neutral", "mild_positive", "strong_positive", "off_topic"))),
        "score_by_class": dict(zip(CHOICES, (*SCORE_LEVELS, None))),
        "fingerprint_fields": FINGERPRINT_FIELDS, "input_hash_version": INPUT_HASH_VERSION,
        "raw_context_sha256": sha256(source_context(stories)), "protocol": protocol,
        "teacher": {"revision": settings["revision"], "model_name": settings.get("model_name"),
                    "model_revision": settings.get("model_revision"), "protocol": settings["protocol"],
                    "model_asset": {"sha256": asset.sha256, "size_bytes": asset.size_bytes, "source": source}},
        "label_decision": "argmax A–F over actual teacher probabilities; every comment classified, no mention gate",
        "limitation": "Model-generated silver labels for offline training/demo, not independent human gold or an accuracy benchmark.",
    }
    return result


def label_snapshot(config, settings, predict, output_dir):
    """predict(entity, item, aliases) — только контекст; старые метки не передаются."""
    def path(value):
        selected = Path(value).expanduser()
        return selected.resolve() if selected.is_absolute() else (ROOT / selected).resolve()

    full_path, subset_path = path(config["snapshot_path"]), path(config["subset_path"])
    assert full_path != subset_path
    metadata_path, subset_metadata_path = full_path.with_suffix(".metadata.json"), subset_path.with_suffix(".metadata.json")
    source_files = {selected: selected.read_bytes() if selected.exists() else None
                    for selected in (full_path, subset_path, metadata_path, subset_metadata_path)}
    full, subset = json.loads(source_files[full_path]), json.loads(source_files[subset_path])
    full_rows = {row["id"]: row for story in full for row in story["comments"]}
    subset_rows = [row for story in subset for row in story["comments"]]
    assert len(full_rows) == sum(len(story["comments"]) for story in full) == 10000
    assert len(subset_rows) == len({row["id"] for row in subset_rows}) == 1000
    assert all(row["id"] in full_rows and source_context([{"comments": [row]}]) ==
               source_context([{"comments": [full_rows[row["id"]]]}]) for row in subset_rows)
    monitoring = {}
    if config.get("entity") is None or config.get("aliases") is None:
        from lib.config import load_project_config
        monitoring = load_project_config()["monitoring"]
    entity = str(config.get("entity") or monitoring["query"]).strip()
    aliases = config.get("aliases")
    if aliases is None:
        aliases = list(monitoring.get("aliases", [])) if entity.casefold() == monitoring.get("query", "").casefold() else []
    assert entity and isinstance(aliases, list) and all(isinstance(alias, str) and alias.strip() for alias in aliases)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labelled, protocol, cached = annotate(full, entity, aliases, predict, settings, output_dir / "demo_labels.sqlite3")
    labelled_rows = {row["id"]: row for story in labelled for row in story["comments"]}
    selected = copy.deepcopy(subset)
    for story in selected:
        story["comments"] = [copy.deepcopy(labelled_rows[row["id"]]) for row in story["comments"]]
    assert source_context(selected) == source_context(subset)
    assert all(row == labelled_rows[row["id"]] for story in selected for row in story["comments"])
    raw, subset_raw = json_bytes(labelled), json_bytes(selected)
    meta = metadata(json.loads(source_files[metadata_path] or b"{}"), labelled, raw, entity, aliases, settings, protocol)
    subset_meta = metadata(json.loads(source_files[subset_metadata_path] or b"{}"), selected, subset_raw, entity, aliases, settings, protocol)
    subset_meta["subset_of"] = {"file": full_path.name, "sha256": sha256(raw), "selection_protocol": "hn-fixed-id-subset-v1",
                                 "method": "Retain existing1000comment IDs and ordering; no new sampling or class stratification after relabeling."}
    meta["subsets"] = [{"file": subset_path.name, "sha256": sha256(subset_raw), "comments": 1000}]
    payloads = {full_path: raw, subset_path: subset_raw, subset_metadata_path: json_bytes(subset_meta), metadata_path: json_bytes(meta)}
    # All four complete files are staged outside source code. The parent manifest
    # is the final commit marker; cross-file replacement is not a filesystem transaction.
    with tempfile.TemporaryDirectory(prefix="hn-labels-", dir=output_dir) as temporary:
        staged = {}
        for index, (destination, payload) in enumerate(payloads.items()):
            candidate = Path(temporary) / str(index)
            with candidate.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            assert candidate.stat().st_dev == destination.parent.stat().st_dev, "Для atomic replace нужен один файловый том"
            staged[destination] = candidate
        for selected_path, original in source_files.items():
            assert (selected_path.read_bytes() if selected_path.exists() else None) == original, "Исходный snapshot изменен другим процессом"
        for destination, candidate in staged.items():
            os.replace(candidate, destination)
    LOGGER.info("Опубликованы 10000 и 1000 полностью размеченных комментариев; из кеша %d", cached)
    return {"snapshot": str(full_path), "sha256": sha256(raw), "subset": str(subset_path),
            "subset_sha256": sha256(subset_raw), "items": 10000, "subset_items": 1000,
            "cached_items": cached, "teacher_revision": settings["revision"]}
