"""Registry selection and benchmark input isolation without neural inference."""

import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo" / "quality"))

import student_evaluation as evaluation
from registry import Registry, serialize


class StudentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "model-registry.sqlite3"
        self.config = {"student": {"registry_path": str(self.path), "entity": "ChatGPT", "model_id": None, "version": None}}

    def registry(self):
        registry = Registry(self.path)
        models = {
            "older": {"id": "older", "entity": "ChatGPT", "kind": "encoder", "version": 1,
                      "updated_at": "2026-11-10", "versions": [{"version": 1, "trained_at": "2026-11-01"}]},
            "latest": {"id": "latest", "entity": "ChatGPT", "kind": "encoder", "version": 2,
                       "versions": [{"version": 1, "trained_at": "2026-11-02"}, {"version": 2, "trained_at": "2026-11-03"}]},
            "untrained": {"id": "untrained", "entity": "ChatGPT", "kind": "encoder", "version": None},
        }
        state = {"models": models, "jobs": {}}
        for model_id, version in (("older", 1), ("latest", 1), ("latest", 2)):
            artifact = {"kind": "encoder", "entity": "ChatGPT", "encoder": {"device": "cpu", "cache_dir": "unused"},
                        "input_format": "target-comment-parent-title-v1", "decision_rule": "marginal-relevance-p-f-below-half-v1",
                        "score_statistic": "mean", "state": {"fixture_version": version}}
            payload, checksum = serialize(artifact)
            registry.save(state, (model_id, version, checksum, payload))
        return registry

    def test_missing_registry_is_not_created(self):
        with self.assertRaisesRegex(AssertionError, "registry"):
            evaluation.load_student(self.config)
        self.assertFalse(self.path.exists())

    def test_latest_trained_version_and_real_source_checksums(self):
        self.registry()
        protocol, _ = evaluation.load_student(self.config)
        self.assertEqual((protocol["model_id"], protocol["version"]), ("latest", 2))
        self.assertEqual(protocol["source_sha256"]["runtime/ml/student.py"],
                         hashlib.sha256((ROOT / "runtime/ml/student.py").read_bytes()).hexdigest())

    def test_gold_fields_are_removed_and_row_entity_is_preserved(self):
        self.registry()
        _, predict = evaluation.load_student(self.config)
        seen = []

        def classify(_artifact, items):
            seen.extend(items)
            return [[0, 0, 1, 0, 0, 0]]

        with patch.object(evaluation.student, "predict", side_effect=classify):
            result = predict("United Airlines", {"id": "row", "story_id": "s", "text": "raw comment",
                             "gold_polarity": "positive", "human_label": 4, "split": "test", "annotation_confidence": 1})
        self.assertEqual(result, [0, 0, 1, 0, 0, 0])
        self.assertEqual(seen[0]["entity"], "United Airlines")
        self.assertEqual(set(seen[0]), {"entity", "id", "story_id", "text", "story_title", "parent_text", "parent_id"})

    def test_explicit_version_and_device_do_not_change_saved_artifact(self):
        registry = self.registry()
        self.config["student"].update(model_id="latest", version=1, device="auto")
        protocol, _ = evaluation.load_student(self.config)
        self.assertEqual((protocol["version"], protocol["device"]), (1, "auto"))
        self.assertEqual(registry.load("latest", 1)["encoder"]["device"], "cpu")

    def test_corrupted_artifact_is_rejected_by_registry(self):
        self.registry()
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE model_artifacts SET payload=? WHERE model_id=? AND version=?", (b"changed", "latest", 2))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            evaluation.load_student(self.config)
