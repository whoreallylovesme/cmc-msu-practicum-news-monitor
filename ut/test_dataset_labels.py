"""Prepared distillation targets must retain real probabilities and raw context."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime/ml")]

from dataset_labels import fingerprint, load_dataset_snapshot, supervision_hash
from lib.teacher_contracts import input_hash


class PreparedLabelsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "snapshot.json"
        self.row = {
            "id": "comment", "story_id": "thread", "parent_id": "parent",
            "story_title": "Discussion", "parent_text": "Context " * 1000,
            "text": "A useful answer", "label_entity": "ChatGPT", "label": "D",
            "label_source": "gguf-pinned-model-and-protocol",
            "label_probabilities": [0.01, 0.02, 0.08, 0.75, 0.12, 0.02],
        }
        self.aliases = ["Chat GPT"]
        self.row["label_fingerprint"] = fingerprint(self.row)
        self.row["label_input_hash"] = input_hash("ChatGPT", self.row, self.aliases)

    def load(self, row, aliases=None):
        self.path.write_text(json.dumps([{"comments": [row]}]))
        return load_dataset_snapshot(self.path, "ChatGPT", aliases=aliases)

    def test_current_aliases_preserve_the_actual_teacher_distribution(self):
        snapshot = self.load(self.row, aliases=self.aliases)
        self.assertEqual(snapshot["labels"]["comment"]["probabilities"], self.row["label_probabilities"])
        self.assertEqual(snapshot["labels"]["comment"]["teacher_input_hash"], self.row["label_input_hash"])

    def test_alias_policy_changes_supervision_even_when_probabilities_are_identical(self):
        original = self.load(self.row, aliases=self.aliases)["labels"]["comment"]
        changed_row = {**self.row, "label_input_hash": input_hash("ChatGPT", self.row, ["GPT chat"])}
        changed = self.load(changed_row, aliases=["GPT chat"])["labels"]["comment"]
        self.assertEqual(original["probabilities"], changed["probabilities"])
        self.assertNotEqual(supervision_hash(original), supervision_hash(changed))
        self.assertEqual(supervision_hash(original), supervision_hash({**original, "unrelated_metadata": "new"}))
        legacy = {"probabilities": original["probabilities"]}
        self.assertEqual(supervision_hash(legacy), supervision_hash({**legacy, "teacher_input_hash": None}))

    def test_changed_aliases_reject_unchanged_comment_targets(self):
        with self.assertRaisesRegex(ValueError, "aliases.*relabel"):
            self.load(self.row, aliases=["GPT chat"])

    def test_changed_id_is_rejected_even_when_raw_fingerprint_is_unchanged(self):
        changed = {**self.row, "id": "different-comment"}
        self.assertEqual(fingerprint(changed), self.row["label_fingerprint"])
        with self.assertRaisesRegex(ValueError, "ID.*relabel"):
            self.load(changed, aliases=self.aliases)

    def test_explicit_policy_requires_hash_but_structural_default_uses_no_global_config(self):
        changed = dict(self.row)
        changed.pop("label_input_hash")
        self.assertIn("comment", self.load(changed)["labels"])
        with self.assertRaisesRegex(ValueError, "relabel"):
            self.load(changed, aliases=[])

    def test_soft_teacher_targets_are_not_replaced_by_label_smoothing(self):
        snapshot = self.load(self.row)
        self.assertEqual(snapshot["labels"]["comment"]["probabilities"], self.row["label_probabilities"])
        changed = copy.deepcopy(self.row)
        changed["label_probabilities"] = [0.01, 0.02, 0.1, 0.73, 0.12, 0.02]
        self.assertNotEqual(snapshot["namespace"], self.load(changed)["namespace"])

    def test_edit_beyond_model_context_limit_invalidates_labels(self):
        changed = copy.deepcopy(self.row)
        changed["parent_text"] += "changed suffix"
        with self.assertRaisesRegex(ValueError, "Stale prepared label"):
            self.load(changed)

    def test_invalid_or_inconsistent_distribution_is_rejected(self):
        for values in (None, [1.0], [0.0] * 6, [1.0, 0, 0, 0, 0, 0], [float("nan")] * 6):
            with self.subTest(values=values):
                changed = {**self.row, "label_probabilities": values}
                with self.assertRaisesRegex(ValueError, "Invalid teacher distribution"):
                    self.load(changed)

    def test_labels_cannot_be_reused_for_another_entity(self):
        self.load(self.row)
        with self.assertRaisesRegex(ValueError, "not 'Another entity'"):
            load_dataset_snapshot(self.path, "Another entity")
