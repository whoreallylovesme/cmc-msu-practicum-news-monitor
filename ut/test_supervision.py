"""Human supervision stays pinned, train-only and separate from model inputs."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.supervision import canonical_target_text, load_auxiliary


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "human_train.jsonl"
        self.rows = [
            {"id": str(index), "group_id": "source:" + str(index // 2), "entity": "ChatGPT",
             "text": f"ChatGPT comment {index}", "target_start": 0, "target_end": 7,
             "split": "train", "gold_relevant": True,
             "gold_polarity": ("negative", "neutral", "positive")[index % 3],
             "annotation_confidence": 0.99}
            for index in range(20)
        ]

    def configured(self, rows=None):
        content = "".join(json.dumps(row) + "\n" for row in (self.rows if rows is None else rows)).encode()
        self.path.write_bytes(content)
        return {"auxiliary_supervision": {
            "path": str(self.path), "sha256": hashlib.sha256(content).hexdigest(), "incremental_replay_items": 5,
        }}

    def test_disabled_auxiliary_reads_no_source(self):
        self.assertEqual(load_auxiliary({"auxiliary_supervision": {"path": None}}),
                         ([], [], {"enabled": False, "items": 0}))

    def test_empty_additional_sources_preserve_the_news_snapshot(self):
        config = self.configured()
        before = load_auxiliary(config)
        config["auxiliary_supervision"]["additional_sources"] = []
        self.assertEqual(load_auxiliary(config), before)
        self.assertEqual(before[2]["sha256"], config["auxiliary_supervision"]["sha256"])

    def test_gold_is_not_a_model_input_and_intensity_is_not_assigned(self):
        items, targets, metadata = load_auxiliary(self.configured())
        self.assertEqual(len(items), 20)
        self.assertEqual(metadata["items"], 20)
        self.assertEqual(metadata["selected_ids"], [item["id"] for item in items])
        self.assertEqual(items[0]["text"], "[TARGET]ChatGPT[/TARGET] comment 0")
        self.assertTrue(all(item["entity"] == "ChatGPT" and item["polarity_only"] for item in items))
        self.assertTrue(all(set(item) == {"id", "story_id", "entity", "text", "parent_text", "story_title", "parent_id", "polarity_only"} for item in items))
        self.assertEqual(targets[0], [0, 1, 0, 0, 0, 0])
        self.assertTrue(all(len(target) == 6 and sum(target) == 1 and target[5] == 0 for target in targets))

    def test_incremental_replay_is_bounded_and_reproducible(self):
        config = self.configured()
        first = load_auxiliary(config, mode="incremental", version=2, seed=42)
        repeated = load_auxiliary(config, mode="incremental", version=2, seed=42)
        later = load_auxiliary(config, mode="incremental", version=3, seed=42)
        self.assertEqual(first, repeated)
        self.assertEqual(len(first[0]), 5)
        self.assertNotEqual(first[2]["selected_ids"], later[2]["selected_ids"])
        self.assertEqual(first[2]["available_items"], 20)

    def test_source_tags_are_only_added_when_training_balance_is_enabled(self):
        self.rows[0]["dataset"] = "newsmtsc_rw"
        self.rows[1]["dataset"] = "mad_tsc"
        config = self.configured()
        items, targets, metadata = load_auxiliary(config)
        config["student_encoder"] = {"training_balance": {}}
        tagged, tagged_targets, tagged_metadata = load_auxiliary(config)
        self.assertEqual([item["training_source"] for item in tagged[:3]],
                         ["newsmtsc_rw", "mad_tsc", "news"])
        self.assertEqual([{key: value for key, value in item.items() if key != "training_source"}
                          for item in tagged], items)
        self.assertEqual(tagged_targets, targets)
        self.assertEqual(tagged_metadata, metadata)

    def test_changed_source_fails_its_checksum(self):
        config = self.configured()
        self.path.write_text(self.path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_auxiliary(config)

    def test_evaluation_rows_unknown_labels_and_stale_spans_are_rejected(self):
        for change in ({"split": "test"}, {"split": "dev"}, {"gold_relevant": False},
                       {"gold_polarity": "unknown"}, {"target_end": 5}, {"target_start": None}):
            with self.subTest(change=change):
                rows = copy.deepcopy(self.rows)
                rows[0].update(change)
                with self.assertRaises(ValueError):
                    load_auxiliary(self.configured(rows))

    def test_markers_use_only_the_provided_span(self):
        row = {"text": "ChatGPT and ChatGPT", "entity": "ChatGPT", "target_start": 12, "target_end": 19}
        self.assertEqual(canonical_target_text(row), "ChatGPT and [TARGET]ChatGPT[/TARGET]")
        self.assertEqual(canonical_target_text(row, False), row["text"])
