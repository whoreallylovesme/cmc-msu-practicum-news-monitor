"""Независимые метрики, групповые интервалы и кеш ответов без передачи gold модели."""

import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sqlite3
import time

from lib.supervision import canonical_target_text


LOGGER = logging.getLogger(__name__)
POLARITIES = ("negative", "neutral", "positive")
INPUT_VERSION = "human-benchmark-input-v1"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def canonical_input(row, mark_target=False):
    """Explicit allowlist: gold, split and annotation confidence never reach the model.

    Optional [TARGET] markers encode human-provided target span, which is task
    input, not its sentiment. No span is guessed for HN or weather examples.
    """
    text = canonical_target_text(row, mark_target)
    comment_id = row["id"]
    if len(comment_id) > 128:
        comment_id = "benchmark-" + hashlib.sha256(comment_id.encode()).hexdigest()
    item = {"id": comment_id, "story_id": row["group_id"], "text": text,
            "story_title": row.get("story_title") or "", "parent_text": row.get("parent_text") or "",
            "parent_id": None}
    return row["entity"], item


def input_hash(entity, item):
    raw = json.dumps({"version": INPUT_VERSION, "entity": entity, "item": item},
                     ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def normalize(values):
    values = [float(value) for value in values]
    assert len(values) == 6 and all(math.isfinite(value) and value >= 0 for value in values)
    total = sum(values)
    assert total > 0, "Model returned an empty probability distribution"
    return [value / total for value in values]


def label_with(fn, rows, cache, revision, mark_target=False, progress_every=20, protocol=None):
    """fn(entity: str, item: dict) -> six probabilities A,B,C,D,E,F.

    SQLite stores model outputs only. Cache identity includes the exact revision
    (including prompt/protocol) and complete projected input, never gold labels.
    Commit after each completed row so interrupted inference resumes safely.
    """
    assert revision and isinstance(revision, str)
    identity = {"input_version": INPUT_VERSION, "mark_target": mark_target, "model_protocol": protocol}
    protocol_hash = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(cache), timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""CREATE TABLE IF NOT EXISTS benchmark_predictions (
        revision TEXT NOT NULL, protocol_hash TEXT NOT NULL, input_hash TEXT NOT NULL, probabilities TEXT NOT NULL,
        inference_seconds REAL NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (revision, protocol_hash, input_hash))""")
    records, started = [], time.perf_counter()
    try:
        for index, row in enumerate(rows):
            entity, item = canonical_input(row, mark_target)
            digest = input_hash(entity, item)
            saved = connection.execute("SELECT probabilities, inference_seconds FROM benchmark_predictions WHERE revision=? AND protocol_hash=? AND input_hash=?",
                                       (revision, protocol_hash, digest)).fetchone()
            if saved is None:
                before = time.perf_counter()
                probabilities = normalize(fn(entity, item))
                elapsed = time.perf_counter() - before
                connection.execute("INSERT INTO benchmark_predictions (revision, protocol_hash, input_hash, probabilities, inference_seconds) VALUES (?, ?, ?, ?, ?)",
                                   (revision, protocol_hash, digest, json.dumps(probabilities), elapsed))
                connection.commit()
            else:
                probabilities, elapsed = normalize(json.loads(saved[0])), float(saved[1])
            records.append({"id": row["id"], "dataset": row["dataset"], "split": row["split"],
                            "revision": revision, "input_hash": digest, "probabilities": probabilities,
                            "protocol_sha256": protocol_hash,
                            "cached": saved is not None, "inference_seconds": elapsed})
            if progress_every and ((index + 1) % progress_every == 0 or index + 1 == len(rows)):
                LOGGER.info("%s: %d/%d examples, elapsed %.1fs, cached %d", revision, index + 1, len(rows),
                            time.perf_counter() - started, sum(record["cached"] for record in records))
    finally:
        connection.close()
    return records


def _confusion(expected, predicted, size):
    matrix = [[0] * size for _ in range(size)]
    for actual, observed in zip(expected, predicted):
        matrix[actual][observed] += 1
    return matrix


def _class_scores(matrix, label):
    correct = matrix[label][label]
    support = sum(matrix[label])
    predicted = sum(row[label] for row in matrix)
    precision = correct / predicted if predicted else 0.0
    recall = correct / support if support else 0.0
    f1 = 2 * correct / (support + predicted) if support + predicted else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def _point_metrics(cases):
    known_polarity = [case for case in cases if case["gold_relevant"] is True and case["gold_polarity"] in POLARITIES]
    polarity_expected = [POLARITIES.index(case["gold_polarity"]) for case in known_polarity]
    # F rejection remains an error for true relevant examples; never drop it.
    polarity_predicted = [case["prediction"] for case in known_polarity]
    polarity_matrix = _confusion(polarity_expected, polarity_predicted, 4)
    known_relevance = [case for case in cases if case["gold_relevant"] is not None]
    relevance_expected = [int(case["gold_relevant"]) for case in known_relevance]
    relevance_predicted = [int(case["prediction"] != 3) for case in known_relevance]
    relevance_matrix = _confusion(relevance_expected, relevance_predicted, 2)
    joint = [case for case in cases if case["gold_relevant"] is False
             or case["gold_relevant"] is True and case["gold_polarity"] in POLARITIES]
    joint_expected = [3 if case["gold_relevant"] is False else POLARITIES.index(case["gold_polarity"]) for case in joint]
    joint_predicted = [case["prediction"] for case in joint]
    joint_matrix = _confusion(joint_expected, joint_predicted, 4)
    present = sorted(set(joint_expected))
    rejection_errors = sum(case["prediction"] == 3 for case in known_polarity)
    positive_relevance = _class_scores(relevance_matrix, 1)
    return {
        "items": len(cases), "relevant_polarity_items": len(known_polarity),
        "polarity_accuracy": sum(a == b for a, b in zip(polarity_expected, polarity_predicted)) / len(known_polarity) if known_polarity else None,
        "polarity_macro_f1": sum(_class_scores(polarity_matrix, label)["f1"] for label in range(3)) / 3 if known_polarity else None,
        "polarity_confusion_with_rejection": [row[:] for row in polarity_matrix[:3]],
        "polarity_rejection_errors": rejection_errors,
        "relevance_items": len(known_relevance),
        "relevance_accuracy": sum(a == b for a, b in zip(relevance_expected, relevance_predicted)) / len(known_relevance) if known_relevance else None,
        "relevance_precision": positive_relevance["precision"] if known_relevance else None,
        "relevance_recall": positive_relevance["recall"] if known_relevance else None,
        "relevance_positive_f1": positive_relevance["f1"] if known_relevance else None,
        "relevance_macro_f1": sum(_class_scores(relevance_matrix, label)["f1"] for label in range(2)) / 2 if known_relevance else None,
        "relevance_confusion": relevance_matrix,
        "relevance_has_both_gold_classes": len(set(relevance_expected)) == 2,
        "joint_accuracy": sum(a == b for a, b in zip(joint_expected, joint_predicted)) / len(joint) if joint else None,
        "joint_four_class_macro_f1": sum(_class_scores(joint_matrix, label)["f1"] for label in range(4)) / 4 if joint else None,
        "joint_present_class_macro_f1": sum(_class_scores(joint_matrix, label)["f1"] for label in present) / len(present) if present else None,
        "joint_four_class_applicable": len(present) == 4,
        "joint_confusion": joint_matrix,
        "joint_per_class": {name: _class_scores(joint_matrix, index) for index, name in enumerate((*POLARITIES, "irrelevant"))},
    }


def _percentile(values, quantile):
    values = sorted(values)
    point = (len(values) - 1) * quantile
    low, high = math.floor(point), math.ceil(point)
    return values[low] + (values[high] - values[low]) * (point - low)


def grouped_bootstrap(cases, samples=1000, seed=1729):
    """Resample source groups, keeping correlated mentions/tweets together."""
    grouped = {}
    for case in cases:
        grouped.setdefault(case["group_id"], []).append(case)
    groups = list(grouped.values())
    names = ("polarity_accuracy", "polarity_macro_f1", "relevance_positive_f1", "relevance_macro_f1",
             "joint_accuracy", "joint_four_class_macro_f1", "joint_present_class_macro_f1")
    observations = {name: [] for name in names}
    generator = random.Random(seed)
    for _ in range(samples):
        selected = [case for group in generator.choices(groups, k=len(groups)) for case in group]
        metrics = _point_metrics(selected)
        for name in names:
            if metrics[name] is not None:
                observations[name].append(metrics[name])
    return {"method": "percentile group bootstrap, source groups sampled with replacement",
            "groups": len(groups), "samples": samples, "seed": seed,
            "confidence": 0.95,
            "intervals": {name: [_percentile(values, .025), _percentile(values, .975)] if values else None
                          for name, values in observations.items()}}


def summarize(rows, predictions, bootstrap_samples=1000, seed=1729):
    assert rows and len(rows) == len(predictions)
    assert len({row["dataset"] for row in rows}) == 1, "Report datasets separately; do not pool overlapping RW/MT"
    assert len({row["split"] for row in rows}) == 1, "Development and final test metrics must stay separate"
    assert len({row["revision"] for row in predictions}) == 1, "Model revisions must be evaluated separately"
    assert len({row.get("protocol_sha256") for row in predictions}) == 1, "Input protocols must stay separate"
    cases = []
    for row, prediction in zip(rows, predictions):
        assert row["id"] == prediction["id"]
        probabilities = normalize(prediction["probabilities"])
        relevant = probabilities[5] < .5
        polarity = max(range(3), key=[probabilities[0] + probabilities[1], probabilities[2], probabilities[3] + probabilities[4]].__getitem__)
        cases.append({"id": row["id"], "group_id": row["group_id"], "gold_relevant": row["gold_relevant"],
                      "gold_polarity": row["gold_polarity"], "prediction": polarity if relevant else 3})
    fresh = [prediction["inference_seconds"] for prediction in predictions if not prediction["cached"]]
    report = {
        "dataset": rows[0]["dataset"], "split": rows[0]["split"], "revision": predictions[0]["revision"],
        "protocol_sha256": predictions[0].get("protocol_sha256"),
        "reference_sha256": hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "reference": "independent public human annotations; model outputs are not gold",
        "decision": "relevant iff P(F)<0.5; polarity argmax of [P(A)+P(B),P(C),P(D)+P(E)]; F rejection counts as polarity error",
        "class_order": [*POLARITIES, "irrelevant"], "relevance_class_order": ["irrelevant", "relevant"],
        "metrics": _point_metrics(cases),
        "fine_intensity": {"applicable": False, "accuracy": None, "mae": None,
                           "reason": "These human benchmarks do not provide validated five-level intensity labels."},
        "timing": {"uncached_items": len(fresh), "cached_items": len(predictions) - len(fresh),
                   "warm_inference_seconds": sum(fresh),
                   "ms_per_uncached_item": sum(fresh) * 1000 / len(fresh) if fresh else None,
                   "p50_ms": _percentile(fresh, .5) * 1000 if fresh else None,
                   "p95_ms": _percentile(fresh, .95) * 1000 if fresh else None,
                   "note": "Model callback timings exclude model construction/load, include all per-row model work, and exclude cache hits."},
    }
    if bootstrap_samples:
        report["uncertainty"] = grouped_bootstrap(cases, bootstrap_samples, seed)
    return report


def save_result(path, rows, predictions, bootstrap_samples=1000):
    report = summarize(rows, predictions, bootstrap_samples)
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report
