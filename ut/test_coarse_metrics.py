"""Three-way polarity has no intensity agreement requirement."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime/ml")]

import student


class CoarseMetricTests(unittest.TestCase):
    def test_intensity_slots_collapse_to_the_same_task(self):
        expected = [[0.8, 0, 0.2, 0, 0, 0]]
        actual = [[0, 0.8, 0.2, 0, 0, 0]]
        report = student.coarse_fidelity(expected, actual)
        self.assertEqual(report["class_agreement"], 1)
        self.assertAlmostEqual(report["kl_divergence"], 0)
        self.assertEqual(student.fidelity(expected, actual)["class_agreement"], 0)

    def test_binary_relevance_precedes_conditional_polarity(self):
        values = np.asarray([[0.3, 0.2, 0.1, 0.4], [0.1, 0.1, 0.2, 0.6]])
        self.assertEqual(student.coarse_decisions(values).tolist(), [0, 3])

    def test_coarse_sources_score_zero_half_and_one(self):
        rows = [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
        for row, expected in zip(rows, (0, 0.5, 1)):
            self.assertEqual(student.score_distribution(row), (expected, True))
        report = student.coarse_label_metrics([1, 2, 3], rows)
        self.assertEqual(report["class_accuracy"], 1)
        self.assertEqual(report["score_mae"], 0)
        self.assertIsNone(report["class_macro_f1"])

    def test_evaluation_uses_coarse_teacher_agreement(self):
        artifact = {"probability_model": "coarse-factorized", "decision_rule": "marginal-relevance-p-f-below-half-v1"}
        actual = np.asarray([[1, 0, 0, 0, 0, 0]])
        with patch.object(student, "predict", return_value=actual):
            report = student.evaluate(artifact, [{"id": "1", "story_id": "s", "human_label": 1}],
                                      [[0, 1, 0, 0, 0, 0]])
        self.assertEqual(report["teacher_fidelity"]["class_agreement"], 1)
        self.assertEqual(report["human"]["class_accuracy"], 1)
        self.assertEqual(report["human"]["task"], "three-polarities-and-relevance")
