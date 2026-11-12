"""Canonical auxiliary sources keep context and human or teacher targets intact."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.supervision import load_auxiliary


class CanonicalSupervisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("MONITOR_DATA_DIR", None)
        self.rows = [
            {"id": "human:1", "group_id": "article:1", "entity": "Acme",
             "text": "Raw current paragraph.", "parent_text": "Original complete article.",
             "story_title": "Original title", "split": "train", "partial_labels": True, "relevance_only": True,
             "sample_weight": 0.8, "label_probabilities": [0, 0, 0, 1, 0, 0],
             "gold_polarity": "positive", "original_document_label": "negative"},
            {"id": "silver:2", "group_id": "thread:2", "entity": "Acme",
             "text": "Raw comment.", "parent_text": None, "story_title": "Thread title",
             "split": "train", "partial_labels": False, "sample_weight": 1,
             "label_probabilities": [0.03, 0.1, 0.2, 0.3, 0.07, 0.3]},
        ]

    def config(self, rows=None):
        source = self.root / "train.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in (self.rows if rows is None else rows)))
        return {"data_dir": str(self.root / "cache"), "auxiliary_supervision": {
            "path": None, "additional_sources": [{
                "name": "provided", "format": "canonical_supervision_jsonl",
                "asset": {"path": str(source), "is_local": True,
                          "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "size_bytes": source.stat().st_size},
            }],
        }}

    def test_real_context_and_soft_targets_survive_without_gold_features(self):
        items, targets, metadata = load_auxiliary(self.config())
        self.assertEqual(targets, [row["label_probabilities"] for row in self.rows])
        self.assertEqual(items[0]["text"], self.rows[0]["text"])
        self.assertEqual(items[0]["parent_text"], self.rows[0]["parent_text"])
        self.assertEqual(items[0]["story_title"], self.rows[0]["story_title"])
        self.assertEqual(items[0]["id"], "aux:human:1")
        self.assertEqual(items[0]["story_id"], "aux:article:1")
        self.assertTrue(items[0]["partial_labels"])
        self.assertFalse(items[1]["partial_labels"])
        self.assertTrue(items[0]["relevance_only"])
        self.assertFalse(items[1]["relevance_only"])
        self.assertEqual(items[0]["sample_weight"], 0.8)
        self.assertNotIn("gold_polarity", items[0])
        self.assertNotIn("original_document_label", items[0])
        self.assertEqual(metadata["sources"][0]["selection_sha256"], None)

    def test_eval_rows_duplicates_and_missing_target_cannot_enter_training(self):
        for change in ({"split": "test"}, {"split": "dev"}, {"entity": ""}, {"partial_labels": None}):
            rows = copy.deepcopy(self.rows)
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_auxiliary(self.config(rows))
        with self.assertRaises(ValueError):
            load_auxiliary(self.config([self.rows[0], self.rows[0]]))

    def test_invalid_probabilities_and_confidence_are_rejected(self):
        for change in ({"label_probabilities": [0] * 6}, {"label_probabilities": [1] * 4},
                       {"label_probabilities": [0, 0, -1, 0, 0, 2]},
                       {"label_probabilities": [0, 0, float("nan"), 0, 0, 1]},
                       {"sample_weight": 0}, {"sample_weight": 2}):
            rows = copy.deepcopy(self.rows)
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_auxiliary(self.config(rows))

    def test_source_tags_and_revision_follow_the_pinned_data(self):
        config = self.config()
        before = load_auxiliary(config)
        config["student_encoder"] = {"training_balance": {}}
        tagged = load_auxiliary(config)
        self.assertTrue(all(item["training_source"] == "provided" for item in tagged[0]))
        self.assertEqual(before[1:], tagged[1:])
        changed = copy.deepcopy(self.rows)
        changed[1]["label_probabilities"] = [0.1, 0.1, 0.1, 0.4, 0.1, 0.2]
        self.assertNotEqual(load_auxiliary(self.config(changed))[2]["sha256"], before[2]["sha256"])
