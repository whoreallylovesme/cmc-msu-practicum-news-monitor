"""Evaluate teacher labels and trained students against the saved manual review."""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime" / "ml"))
from lib.config import load_config, load_monitoring
from lib.mentions import MentionMatcher
from student import label_metrics
from teacher import ensure_model, llama_teacher
from lib.client import all_items, call, monitoring_options, wait_ready


def metrics(labels, probabilities, statistic="mean"):
    actual = np.asarray(probabilities).argmax(axis=1)
    return {
        **label_metrics(labels, probabilities, statistic),
        "predicted_classes": dict(sorted(Counter(map(int, actual)).items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--entity")
    parser.add_argument("--reference", type=Path, default=ROOT / "runtime/ml/evaluation.json")
    parser.add_argument("--reference-corpus", type=Path, default=ROOT / "runtime/ml/evaluation_corpus.json")
    parser.add_argument("--baseline-artifact", type=Path, help="Optional trusted local artifact evaluated on the identical audit")
    parser.add_argument("--teacher-labels", type=Path, help="Optional legacy teacher labels for comparison")
    parser.add_argument("--model-id", help="Evaluate one model instead of the latest model of each kind")
    parser.add_argument("--output", type=Path, default=ROOT / ".local-data/evaluation.json")
    parser.add_argument("--benchmark-weights", type=Path, help="Benchmark the live teacher using a local GGUF")
    parser.add_argument("--llama-binary", default=os.environ.get("ML_LLAMA_BINARY", "llama-server"))
    parser.add_argument("--benchmark-items", type=int, default=12)
    parser.add_argument("--benchmark-gpu-layers", type=int, default=0)
    args = parser.parse_args()
    if args.benchmark_items < 1:
        parser.error("benchmark-items must be positive")
    wait_ready(args.url, 30)
    entity, _, _ = monitoring_options(args.url, entity=args.entity)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    teacher = json.loads(args.teacher_labels.read_text(encoding="utf-8")) if args.teacher_labels else None
    if reference["entity"] != entity or (teacher and teacher["entity"] != entity):
        parser.error("reference, teacher labels and configured query must name the same entity")
    comments = {item["id"]: item for item in all_items(args.url, "/api/comments", {"entity": entity})}
    external = {}
    if args.reference_corpus.is_file():
        corpus = json.loads(args.reference_corpus.read_text(encoding="utf-8"))
        if corpus["entity"] != entity:
            parser.error("reference corpus and monitoring query must name the same entity")
        external = {item["id"]: item for item in corpus["items"]}
        comments.update(external)
    reviewed = [item for item in reference["items"] if item["comment_id"] in comments]
    if len(reviewed) != len(reference["items"]):
        parser.error("provide the complete reference corpus or import its comments before evaluating")
    primary = reviewed
    if not primary:
        parser.error("no reference labels")
    labels = {item["comment_id"]: item["probabilities"] for item in teacher["items"]} if teacher else {}
    expected = [item["label"] for item in primary]
    score_statistic = load_config(ROOT / "runtime/ml/config.yaml", env_var="ML_CONFIG").get("score_statistic", "mean")
    monitoring = load_monitoring()
    policy = {"entity": entity, "aliases": monitoring["aliases"], "options": monitoring["matching"]}
    if teacher:
        policy = teacher["teacher"].get("mention_policy", policy)
    matcher = MentionMatcher(entity, policy["aliases"], policy["options"])
    in_scope = [item for item in primary if matcher.match(comments[item["comment_id"]]) is not None]
    if not in_scope:
        parser.error("the reference has no clear entity mentions under the saved matching policy")
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "entity": entity,
        "timing_note": "Saved snapshot timings may use prepared or in-process NLI features; see README for uncached backbone measurements.",
        "reference": {
            "annotator": reference["annotator"],
            "limitation": reference["limitation"],
            "reviewed_comments": len(reviewed),
            "primary_comments": len(primary),
            "in_scope_comments": len(in_scope),
            "scope_excluded_comments": len(primary) - len(in_scope),
            "all_confidence_levels_included": True,
            "low_confidence_comments": sum(item["confidence"] == "low" for item in primary),
            "class_counts": dict(sorted(Counter(expected).items())),
            "scoring_scale": [0, 5],
        },
        "students": [],
    }
    if teacher:
        report["legacy_teacher"] = {
            "provenance": teacher["teacher"],
            "manual_review_all": metrics(expected, [labels[item["comment_id"]] for item in primary], score_statistic),
            "manual_review_in_scope": metrics([item["label"] for item in in_scope], [labels[item["comment_id"]] for item in in_scope], score_statistic),
        }
    seen_kinds = set()
    for model in call(args.url, "/api/models", query={"entity": entity})["items"]:
        if args.model_id and model["id"] != args.model_id:
            continue
        if not model.get("version") or model["kind"] in seen_kinds:
            continue
        summary = call(args.url, f'/api/models/{model["id"]}/summary')
        predictions = {
            item["comment_id"]: item["probabilities"]
            for item in all_items(args.url, "/api/evidence", {"entity": entity, "model_id": model["id"]})
        }
        if external:
            rows = list(external.values())
            for offset in range(0, len(rows), 100):
                result = call(args.url, f'/api/models/{model["id"]}/predict', body={
                    "entity": entity, "version": model["version"], "items": rows[offset:offset + 100],
                })
                predictions.update({item["comment_id"]: item["probabilities"] for item in result["items"]})
        if not all(item["comment_id"] in predictions for item in primary):
            continue
        version = next(row for row in summary["versions"] if row["version"] == model["version"])
        statistic = version.get("score_statistic", "mean")
        held_out = set(version.get("holdout_story_ids", []))
        trained = set(version.get("training_story_ids", []))
        unseen = [item for item in primary if (comments[item["comment_id"]]["story_id"] not in trained
                  if trained else comments[item["comment_id"]]["story_id"] in held_out)]
        scoped_unseen = [item for item in in_scope if item in unseen]
        report["students"].append({
            "model_id": model["id"],
            "version": model["version"],
            "kind": model["kind"],
            "manual_review_all": metrics(expected, [predictions[item["comment_id"]] for item in primary], statistic),
            "manual_review_holdout": metrics(
                [item["label"] for item in unseen],
                [predictions[item["comment_id"]] for item in unseen],
                statistic,
            ) if unseen else None,
            "manual_review_holdout_in_scope": metrics(
                [item["label"] for item in scoped_unseen],
                [predictions[item["comment_id"]] for item in scoped_unseen],
                statistic,
            ) if scoped_unseen else None,
            "holdout_story_ids": sorted(held_out),
            "training_story_ids": sorted(trained),
            "teacher_fidelity": summary["metrics"]["holdout"].get("teacher_fidelity"),
            "teacher_fidelity_candidates": summary["metrics"]["holdout"].get("teacher_fidelity_candidates"),
            "student_ms_per_item": summary["metrics"]["holdout"]["student_ms_per_item"],
            "artifact_bytes": version["artifact_bytes"],
            "output_semantics": "three-polarity sentiment index" if summary["metrics"]["optimization"].get("parameters") == 1 else "joint intensity/relevance",
            "fine_intensity_metrics_applicable": summary["metrics"]["optimization"].get("head") != "temperature",
        })
        seen_kinds.add(model["kind"])
    if not report["students"]:
        parser.error("train and score at least one model before evaluating")
    if args.baseline_artifact:
        import hashlib
        import joblib
        import student

        os.environ.setdefault("ML_ENCODER_CACHE_DIR", str(ROOT / ".local-data/ml/encoder-weights"))
        baseline = joblib.load(args.baseline_artifact)
        probabilities = student.predict(baseline, [comments[item["comment_id"]] for item in primary])
        lookup = {item["comment_id"]: value for item, value in zip(primary, probabilities)}
        report["baseline"] = {
            "artifact_sha256": hashlib.sha256(args.baseline_artifact.read_bytes()).hexdigest(),
            "same_audit_comments": True,
            "manual_review_all": metrics(expected, probabilities, baseline.get("score_statistic", "mean")),
            "manual_review_in_scope": metrics([item["label"] for item in in_scope],
                [lookup[item["comment_id"]] for item in in_scope], baseline.get("score_statistic", "mean")),
        }
    if args.benchmark_weights:
        config = load_config(ROOT / "runtime/ml/config.yaml", env_var="ML_CONFIG")
        config["model_weights"] = {"path": str(args.benchmark_weights.resolve()), "is_local": True}
        config["llama_binary"] = args.llama_binary
        config["llama_gpu_layers"] = args.benchmark_gpu_layers
        digest = ensure_model(config)
        samples = [comments[item["comment_id"]] for item in in_scope[:args.benchmark_items]]
        log_path = args.benchmark_weights.parent / "evaluation-llama.log"
        with llama_teacher(config, log_path) as live:
            started = perf_counter()
            for item in samples:
                live.distribution(entity, item)
            elapsed = perf_counter() - started
        report["optional_llm_benchmark"] = {
            "items": len(samples), "ms_per_item": elapsed * 1000 / len(samples),
            "model_sha256": digest, "includes_server_startup": False,
            "gpu_layers": args.benchmark_gpu_layers, "selected_mentions_only": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
