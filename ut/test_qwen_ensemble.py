import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "ml"))
sys.path.insert(0, str(ROOT))

import qwen_ensemble
import student


class QwenEnsembleTests(unittest.TestCase):
    def test_parse_label_uses_final_class(self):
        self.assertEqual(qwen_ensemble.parse_label("EVIDENCE: Useful.\nCLASS: D"), 3)
        self.assertEqual(qwen_ensemble.parse_label("CLASS: A\nCLASS: F"), 5)
        with self.assertRaisesRegex(ValueError, "no CLASS"):
            qwen_ensemble.parse_label("EVIDENCE: Missing label")

    def test_two_irrelevant_votes_override_only_neutral(self):
        base = np.asarray([2, 2, 1, 3, 5])
        auxiliary = np.asarray([
            [5, 5, 5, 5, 5],
            [5, 1, 5, 1, 5],
            [1, 5, 1, 5, 1],
            [1, 1, 1, 5, 5],
        ])
        actual = qwen_ensemble.combine_labels(base, auxiliary)
        np.testing.assert_array_equal(actual, [5, 5, 1, 3, 5])

    def test_vote_does_not_use_identifiers_or_change_other_classes(self):
        auxiliary = np.zeros((4, 3), dtype=np.int64)
        auxiliary[0, 0] = 5
        actual = qwen_ensemble.combine_labels([0, 2, 4], auxiliary)
        np.testing.assert_array_equal(actual, [0, 2, 4])

    def test_invalid_vote_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "four auxiliary"):
            qwen_ensemble.combine_labels([2], np.zeros((3, 1), dtype=np.int64))

    def test_student_dispatches_qwen_artifact(self):
        artifact = {"kind": "encoder", "architecture": qwen_ensemble.PROTOCOL}
        expected = np.eye(6)[[5]]
        item = {"id": "one", "story_id": "story", "text": "Unrelated"}
        with patch("qwen_ensemble.predict", return_value=expected) as predict:
            actual = student.predict(artifact, [item])
        predict.assert_called_once_with(artifact, [item])
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
