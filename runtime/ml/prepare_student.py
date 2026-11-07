"""Train and bundle a portable student from reviewed labels or a teacher snapshot."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime/ml")]

from lib.config import load_config, load_monitoring, load_project_config
from registry import serialize
from reviewed_labels import apply_reviewed, fingerprint, load_reviewed_snapshot, supervision_hash
from teacher import input_fingerprint, load_snapshot, validate_snapshot_policy
import student
import training_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(load_project_config()["path_to_offline_demo_data"]))
    parser.add_argument("--teacher-labels", type=Path, default=ROOT / "runtime/ml/hn_snapshot_labels.json")
    parser.add_argument("--labels-source", choices=["reviewed", "snapshot"], default="reviewed")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/ml/pretrained")
    parser.add_argument("--reference", type=Path, default=ROOT / "runtime/ml/evaluation.json")
    parser.add_argument("--reference-corpus", type=Path, default=ROOT / "runtime/ml/evaluation_corpus.json")
    parser.add_argument("--kind", choices=["semantic", "softmax", "hard_label", "tiny_mlp"], default="semantic")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.teacher_labels = args.teacher_labels.resolve()
    args.output = args.output.resolve()
    args.reference = args.reference.resolve()
    args.reference_corpus = args.reference_corpus.resolve()
    monitoring = load_monitoring()
    entity = monitoring["query"]
    config = load_config(ROOT / "runtime/ml/config.yaml", env_var="ML_CONFIG")
    config["monitoring"] = monitoring
    config.update(semantic_entity=entity, semantic_aliases=monitoring["aliases"], semantic_matching=monitoring["matching"])
    # Match the service's working directory for configured relative asset paths.
    os.chdir(ROOT / "runtime/ml")
    items = [item for story in json.loads(args.dataset.read_text(encoding="utf-8")) for item in story["comments"]]
    plan = training_data.from_scratch(items, len(items), config["validation_fraction"])
    train, holdout = plan["train"], plan["holdout"]
    review_path = config.get("reviewed_training_path")
    if args.labels_source == "snapshot":
        snapshot = load_snapshot(args.teacher_labels, entity)
        validate_snapshot_policy(snapshot, config, entity)
        for item in items:
            label = snapshot["labels"].get(str(item["id"]))
            if not label or label["input_hash"] != input_fingerprint(snapshot["entity"], item):
                raise ValueError(f"Missing or stale teacher label for {item['id']}")
        targets = [snapshot["labels"][str(item["id"])]["probabilities"] for item in train]
        holdout_targets = [snapshot["labels"][str(item["id"])]["probabilities"] for item in holdout]
    else:
        if not review_path:
            raise ValueError("Set reviewed_training_path before preparing a reviewed model")
        snapshot = load_reviewed_snapshot(ROOT / "runtime/ml" / review_path, entity)
        policy = {"entity": entity, "aliases": monitoring["aliases"], "options": monitoring["matching"]}
        candidates = set(student.candidate_indices({"mention_policy": policy}, train))
        targets = []
        for index, item in enumerate(train):
            if index not in candidates:
                targets.append([0.0] * 5 + [1.0])
                continue
            row = snapshot["labels"].get(str(item["id"]))
            if not row or row["fingerprint"] != fingerprint(item):
                raise ValueError(f"Missing or stale reviewed training label for {item['id']}")
            item["supervision_hash"] = supervision_hash(row)
            targets.append(row["probabilities"])
        holdout_targets = None
        plan["history"] = training_data.history(None, train, holdout, plan["history"]["roles"])
    fit_items, fit_targets, review = apply_reviewed(
        train, targets, entity, ROOT / "runtime/ml" / review_path if review_path else None,
        int(config.get("reviewed_training_weight", 3)),
    )
    artifact, optimization = student.fit(fit_items, fit_targets, args.kind, config)
    artifact["training_history"] = plan["history"]
    validation = student.evaluate(artifact, holdout, holdout_targets)
    payload, checksum = serialize(artifact)
    now = datetime.now(timezone.utc).isoformat()
    metrics = {
        "mode": "from_scratch", "base_version": None,
        "training_items": len(train), "training_threads": len({item["story_id"] for item in train}),
        "holdout": validation, "optimization": optimization, **plan["counts"],
        "label_source": snapshot["teacher"],
        "reviewed_training": review,
    }
    if args.reference.is_file():
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if reference["entity"].casefold() == entity.casefold():
            held_out = {str(item["id"]): item for item in holdout}
            if args.reference_corpus.is_file():
                corpus = json.loads(args.reference_corpus.read_text(encoding="utf-8"))
                extra = corpus["items"]
                trained_stories = {item["story_id"] for item in train}
                if any(item["story_id"] in trained_stories for item in extra):
                    raise ValueError("External evaluation corpus overlaps training threads")
                held_out.update({str(item["id"]): item for item in extra})
            all_reviewed = [row for row in reference["items"] if row["comment_id"] in held_out]
            covered = student.candidate_indices(artifact, [held_out[row["comment_id"]] for row in all_reviewed])
            reviewed = [all_reviewed[index] for index in covered]
            if reviewed:
                expected = [row["label"] for row in reviewed]
                metrics["assistant_reference"] = {
                    "annotator": reference["annotator"], "limitation": reference["limitation"],
                    "scope": "Held-out threads only; these labels never enter training",
                    "reviewed_comments": len(all_reviewed), "in_scope_comments": len(reviewed),
                    "scope_excluded_comments": len(all_reviewed) - len(reviewed),
                    "all_confidence_levels_included": True,
                    "output_semantics": "three-polarity sentiment index" if artifact.get("head") == "temperature" else "five intensity classes",
                    "fine_intensity_metrics_applicable": artifact.get("head") != "temperature",
                    "student": student.label_metrics(expected, student.predict(artifact, [held_out[row["comment_id"]] for row in reviewed]), artifact.get("score_statistic", "mean")),
                }
                if args.labels_source == "snapshot" and all(row["comment_id"] in snapshot["labels"] for row in reviewed):
                    metrics["assistant_reference"]["teacher"] = student.label_metrics(expected, [snapshot["labels"][row["comment_id"]]["probabilities"] for row in reviewed], artifact.get("score_statistic", "mean"))
    record = {
        "version": 1, "teacher": args.labels_source, "teacher_revision": snapshot["namespace"],
        "mode": "from_scratch", "base_version": None, "trained_at": now, "metrics": metrics,
        "holdout_story_ids": sorted({item["story_id"] for item in holdout}),
        "training_story_ids": sorted({item["story_id"] for item in train}),
        "artifact_bytes": len(payload), "artifact_sha256": checksum,
        "mention_policy": artifact.get("mention_policy"),
        "score_statistic": artifact.get("score_statistic", "mean"),
    }
    model = {
        "id": "bundled-" + checksum[:20], "entity": entity, "kind": args.kind,
        "status": "ready", "version": 1, "teacher": args.labels_source, "mode": "from_scratch",
        "metrics": metrics, "created_at": now, "updated_at": now, "last_error": None,
        "versions": [record], "bundled": True,
        "mention_policy": artifact.get("mention_policy"),
        "score_statistic": artifact.get("score_statistic", "mean"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output / "student.joblib"
    artifact_path.with_suffix(".tmp").write_bytes(payload)
    artifact_path.with_suffix(".tmp").replace(artifact_path)
    manifest = args.output / "manifest.json"
    manifest.with_suffix(".tmp").write_text(json.dumps(
        {"artifact": artifact_path.name, "model": model}, ensure_ascii=False, indent=2,
    ) + "\n", encoding="utf-8")
    manifest.with_suffix(".tmp").replace(manifest)
    print(json.dumps({"model_id": model["id"], "artifact_bytes": len(payload), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
