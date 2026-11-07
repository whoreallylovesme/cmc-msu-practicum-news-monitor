"""Verify the prelabelled Hacker News demo through the real five-service API."""

import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.client import all_items, assert_status, call, monitoring_options, train_model, wait_job, wait_ready
from lib.cli import configure_logging, load_cli_config, write_json
from lib.mentions import MentionMatcher


DATASETS = {"1k": 1000, "10k": 10000}
FINGERPRINT_FIELDS = ("text", "parent_text", "story_title", "story_id", "parent_id")
LOGGER = logging.getLogger(__name__)


def context_fingerprint(item):
    fields = {key: item.get(key) for key in FINGERPRINT_FIELDS}
    raw = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_dataset(document, expected_items):
    assert isinstance(document, list), "The demo dataset must contain a list of Hacker News stories"
    assert all(
        isinstance(story, dict) and isinstance(story.get("comments"), list) for story in document
    ), "Every demo story must contain a comments list"
    rows = [row for story in document for row in story["comments"]]
    assert len(rows) == expected_items, f"Expected {expected_items} labelled comments, found {len(rows)}"
    ids = set()
    entities = set()
    sources = Counter()
    classes = Counter()
    for row in rows:
        assert isinstance(row, dict), "A dataset comment is not an object"
        cid = row.get("id")
        assert isinstance(cid, str) and cid and cid not in ids, "Dataset comment IDs must be nonempty and unique"
        ids.add(cid)
        label = row.get("label")
        assert isinstance(label, str) and label in tuple("ABCDEF"), f"Missing or invalid A-F label for {cid}"
        entity = row.get("label_entity")
        assert isinstance(entity, str) and entity.strip(), f"Missing label_entity for {cid}"
        entities.add(entity.strip())
        source = row.get("label_source")
        assert isinstance(source, str) and source.strip(), f"Missing label_source for {cid}"
        assert row.get("label_fingerprint") == context_fingerprint(row), f"Stale dataset label context for {cid}"
        sources[source] += 1
        classes[label] += 1
    assert len(entities) == 1, "Demo labels must refer to exactly one entity"
    return {"items": rows, "entity": entities.pop(), "sources": dict(sources), "classes": dict(classes)}


def load_dataset(name):
    path = ROOT / f"demo/hn_{name}.json"
    return validate_dataset(json.loads(path.read_text(encoding="utf-8")), DATASETS[name])


def validate_prediction(prediction, comment_id, scale):
    assert prediction.get("comment_id") == comment_id, "Prediction returned an unexpected comment ID"
    values = prediction.get("probabilities")
    assert (
        isinstance(values, list) and len(values) == 6
        and all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in values)
    ), "Prediction requires six finite nonnegative probabilities"
    assert math.isclose(sum(values), 1, abs_tol=1e-5), "Prediction probabilities do not sum to one"
    score = prediction.get("score")
    assert (
        isinstance(score, (int, float)) and math.isfinite(score)
        and scale["min"] <= score <= scale["max"]
    ), "New in-scope text did not receive a finite public-scale score"


def main(argv=None):
    _, config = load_cli_config(ROOT / "demo/1k.yaml", argv)
    args = SimpleNamespace(**config["check"], dataset=config["dataset"])
    args.url = os.environ.get("MONITOR_E2E_URL", args.url)
    if args.pages < 1 or not 1 <= args.batch_size <= 100:
        raise ValueError("Use at least one page and a batch size between 1 and 100")
    if args.source != "hn_snapshot" or args.teacher != "dataset":
        raise ValueError("The prepared demo requires source hn_snapshot and teacher dataset")
    expected_items = DATASETS[args.dataset]
    dataset = load_dataset(args.dataset)
    wait_ready(args.url, args.timeout)
    entity, source, teacher = monitoring_options(args.url, entity=args.entity, source=args.source, teacher=args.teacher)
    assert entity.strip().casefold() == dataset["entity"].casefold(), "Configured entity does not match dataset labels"
    config = call(args.url, "/api/config")
    args.kind = args.kind or config["pipeline"]["model_kind"]
    if args.kind not in {"hard_label", "softmax", "tiny_mlp"}:
        raise ValueError("The prepared demo requires a small classifier: hard_label, softmax or tiny_mlp")
    if args.training_limit is None:
        args.training_limit = config["pipeline"]["training_limit"]
    if not expected_items <= args.training_limit <= 10000:
        raise ValueError(f"The complete dataset demo needs check.training_limit between {expected_items} and 10000")
    monitoring = config["monitoring"]
    aliases = monitoring["aliases"] if entity.casefold() == monitoring["query"].casefold() else []
    matcher = MentionMatcher(entity, aliases, monitoring["matching"])
    expected_mentions = sum(matcher.match(row) is not None for row in dataset["items"])
    scale = config.get("score_scale", {"min": 0, "max": 5})
    batch_request = {"entity": entity, "source": source, "limit": args.batch_size}
    assert_status(args.url, 401, "/api/config", role=None)
    assert_status(args.url, 403, "/api/import", body=batch_request, role="viewer")
    cursor = None
    seen_cursors = set()
    received = 0
    batches = 0
    for _ in range(args.pages):
        result = call(args.url, "/api/import", body={**batch_request, "cursor": cursor})
        assert result["source"] == "hn_snapshot", "Collector returned a different source"
        received += result["received"]
        batches += 1
        cursor = result["next_cursor"]
        if not cursor:
            break
        assert cursor not in seen_cursors, "Collector repeated its continuation cursor"
        seen_cursors.add(cursor)
    assert not cursor, "Snapshot import stopped before the final cursor; increase check.pages"
    assert received == expected_items, f"Expected {expected_items} imported rows, received {received}"
    stored = list(all_items(args.url, "/api/comments", {"entity": entity}))
    ids = [row["id"] for row in stored]
    expected = {row["id"]: row for row in dataset["items"]}
    assert (
        len(ids) == len(set(ids)) == expected_items and set(ids) == set(expected)
    ), "Imported comments are duplicated, missing, or mixed with unrelated entity data"
    assert all(
        context_fingerprint(row) == expected[row["id"]]["label_fingerprint"] for row in stored
    ), "Imported raw context differs from its prelabelled dataset fingerprint"
    kinds = [args.kind]
    if args.compare:
        kinds.append("softmax" if args.kind == "hard_label" else "hard_label")
    results = []
    for kind in kinds:
        model = train_model(args.url, entity, teacher, kind, args.training_limit, args.timeout)
        model_id = model["id"]
        metrics = model.get("metrics") or {}
        holdout = metrics.get("holdout") or {}
        labelled = metrics.get("training_items", 0) + holdout.get("items", 0)
        assert (
            model.get("kind") == kind and model.get("teacher") == "dataset"
        ), "Model did not use the selected small dataset classifier"
        assert (
            labelled == expected_items and metrics.get("labels_from_dataset") == labelled
        ), "Dataset labels must cover every training and held-out row"
        assert metrics.get("teacher_inference_items") == 0, "Dataset demo unexpectedly invoked a teacher model"
        assert not (
            metrics.get("optimization") or {}
        ).get("encoder_frozen"), "The small demo unexpectedly contains a frozen NN encoder"
        scored = call(args.url, "/api/score", body={"model_id": model_id, "entity": entity, "limit": expected_items})
        assert (
            scored["processed"] == scored["stored"] == expected_items
        ), f"The model did not score all {expected_items} comments"
        params = {"entity": entity, "model_id": model_id}
        analytics = call(args.url, "/api/analytics", query=params, role="viewer")
        assert (
            analytics["summary"]["comments"] == expected_items
            and analytics["summary"]["mentions"] == expected_mentions
            and analytics["summary"]["scored"] == expected_items
        ), f"Analytics did not retain {expected_items} comments / {expected_mentions} mentions / {expected_items} predictions"
        periods = analytics["periods"]
        assert (
            periods and sum(row["scored"] for row in periods) == expected_items
        ), "Timeline does not cover the full scored dataset"
        evidence = list(all_items(args.url, "/api/comments", params))
        evidence_ids = [row["comment_id"] for row in evidence]
        assert (
            len(evidence_ids) == len(set(evidence_ids)) == expected_items and set(evidence_ids) == set(expected)
        ), "Evidence omits or duplicates scored comments"
        report = call(args.url, "/api/report", query=params, role="viewer", raw=True)
        assert (
            "<table" in report and "Настроение обсуждений" in report and "{{" not in report
        ), "The rendered HTML report is incomplete"
        nonce = uuid4().hex
        novel = {"id": "demo-new-" + nonce, "story_id": "demo-story-" + nonce,
                 "text": f"{entity} explains useful examples, and I check the answers myself. New input {nonce}."}
        prediction = call(args.url, f"/services/ml/models/{model_id}/predict", body={
            "version": model["version"], "items": [novel],
        })
        assert len(prediction["items"]) == 1, "New-text prediction returned an incorrect number of rows"
        validate_prediction(prediction["items"][0], novel["id"], scale)
        job = call(args.url, f"/services/ml/models/{model_id}/train", body={
            "teacher": "dataset", "mode": "incremental", "limit": args.training_limit,
        })
        skipped = wait_job(args.url, job["job_id"], args.timeout)
        assert (
            skipped["status"] == "skipped" and skipped.get("version") == model["version"]
        ), "Unchanged dataset incremental training must skip without creating a version"
        current = call(args.url, f"/services/ml/models/{model_id}/summary")
        assert (
            current["status"] == "ready" and current["version"] == model["version"]
            and len(current["versions"]) == len(model["versions"])
        ), "Skipped training changed the ready model or version history"
        results.append({"kind": kind, "model_id": model_id, "version": model["version"],
                        "dataset_labels": labelled, "teacher_inference_items": 0,
                        "scored": scored["stored"], "mentions": analytics["summary"]["mentions"],
                        "timeline_periods": len(periods), "evidence": len(evidence),
                        "new_text_score": prediction["items"][0]["score"], "incremental": "skipped"})
    proof = {"status": "passed", "dataset": args.dataset, "entity": entity, "source": "hn_snapshot", "label_source": "dataset",
             "dataset_rows": len(expected), "label_sources": dataset["sources"], "labels": dataset["classes"],
             "expected_mentions": expected_mentions,
             "reference_kind": "prelabelled silver dataset; not independent human gold",
             "imported": received, "import_batches": batches, "unique_stored_ids": len(set(ids)),
             "anonymous_HTTP": 401, "viewer_write_HTTP": 403, "models": results}
    write_json(proof)
    return proof


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except Exception:
        LOGGER.exception("Snapshot E2E failed")
        sys.exit(1)
