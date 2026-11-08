"""Four-logit tone/relevance head and legacy six-logit compatibility."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime/ml")]
import neural
from registry import Registry, serialize
from ut import test_student as student_fixtures

MODE = "coarse-factorized"


class CoarseMathTests(unittest.TestCase):
    def test_legacy_identity_and_flat_probabilities_are_exact(self):
        artifact = {"encoder": {"files": {"model.safetensors": {"sha256": "a" * 64}},
                    "device": "cpu", "dtype": "float32", "max_length": 48, "threads": 1},
                    "lora": {"rank": 2, "alpha": 4, "dropout": 0.0}, "entity": "x", "aliases": [],
                    "input_format": neural.INPUT_FORMAT, "training": {"seed": 42},
                    "state": {"classifier": np.arange(6, dtype=np.float32)}}
        # Recorded with the pre-coarse production implementation, SHA5ea8e16b.
        self.assertEqual(neural._identity(artifact), "ec9dc8cfd10c6044cfc3b833e9260cdb85bb01913f971bcd3084892367c5d96a")
        self.assertEqual(neural._identity(artifact), neural._identity({**artifact, "probability_model": "flat"}))
        logits = torch.tensor([[1., -2, 3, -.5, 0, .3]])
        torch.testing.assert_close(neural._log_probabilities(logits, torch), logits.log_softmax(-1), rtol=0, atol=0)
        for mode in (None, "factorized", "unknown", {}):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                neural._probability_model(mode)

    def test_teacher_and_partial_targets_collapse_without_changing_confidence(self):
        items = [{"id": "teacher"}, {"id": "human", "partial_labels": True, "sample_weight": .3}]
        targets = np.asarray([[.1, .2, .15, .05, .1, .4], [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        before = targets.copy()
        actual, partial, confidence = neural._training_targets(items, targets, MODE)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, [[.3, .15, .15, .4], [1, 0, 0, 0]], atol=1e-7)
        np.testing.assert_array_equal(partial, [False, False])
        np.testing.assert_allclose(confidence, [1, .3], atol=1e-7)
        np.testing.assert_array_equal(targets, before)
        old, old_partial, _ = neural._training_targets(items, targets)
        self.assertEqual(old.shape, (2, 6))
        np.testing.assert_array_equal(old_partial, [False, True])

    def test_balance_uses_one_coarse_group_for_every_source(self):
        items = [{"id": str(i), "training_source": "weather"} for i in range(10)]
        targets = np.asarray([[1., 0, 0, 0]] * 9 + [[0, 0, 0, 1]])
        weights, metadata = neural._balance_weights(items, targets, np.asarray([True, False] * 5),
            {"probability_model": MODE, "training_balance": {}})
        self.assertEqual(len(metadata["groups"]), 1)
        self.assertEqual(metadata["groups"][0]["class_order"], ["negative", "neutral", "positive", "irrelevant"])
        self.assertEqual(metadata["groups"][0]["class_mass"], [9, 0, 0, 1])
        self.assertAlmostEqual(float(weights[-1] / weights[0]), 3, places=6)
        self.assertAlmostEqual(float(weights.mean()), 1, places=6)

    def test_relevance_and_offtopic_gradient_are_separate_from_three_tones(self):
        logits = torch.tensor([[1., -2, 3, .7]], requires_grad=True)
        expected = torch.tensor([[0., 0, 0, 1]])
        probabilities = neural._log_probabilities(logits, torch, MODE).exp()
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))
        torch.testing.assert_close(probabilities[:, 3], logits[:, 3].sigmoid())
        altered = logits.detach().clone()
        altered[:, :3] = torch.tensor([9000., -8000, 7000])
        torch.testing.assert_close(neural._log_probabilities(altered, torch, MODE).exp()[:, 3],
                                   probabilities[:, 3], rtol=0, atol=0)
        loss = neural._loss(logits, expected, torch.tensor([False]), 1, torch, probability_model=MODE)
        gradient, = torch.autograd.grad(loss, logits)
        torch.testing.assert_close(gradient[:, :3], torch.zeros(1, 3), rtol=0, atol=0)
        torch.testing.assert_close(gradient[:, 3], logits[:, 3].sigmoid() - 1)

    def test_soft_joint_kl_gradient_and_joint_temperature(self):
        logits = torch.tensor([[1., -2, 3, .3]], requires_grad=True)
        expected = torch.tensor([[.3, .15, .15, .4]])
        joint = neural._log_probabilities(logits, torch, MODE).exp()
        reference = (expected * (expected.log() - joint.log())).sum()
        loss = neural._loss(logits, expected, torch.tensor([True]), 1, torch, probability_model=MODE)
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(torch.autograd.grad(loss, logits, retain_graph=True)[0],
                                   torch.autograd.grad(reference, logits)[0])
        tempered = neural._log_probabilities(logits, torch, MODE, 2).exp()
        torch.testing.assert_close(tempered, joint.sqrt() / joint.sqrt().sum(-1, keepdim=True))
        self.assertGreater(float((tempered - neural._log_probabilities(logits / 2, torch, MODE).exp()).abs().max().detach()), .01)


class CoarseSnapshotTests(unittest.TestCase):
    def setUp(self):
        student_fixtures.StudentTests.setUp(self)
        self.config["data_dir"] = str(self.root / "data")
        self.config["student_encoder"].update(probability_model=MODE, max_steps=1,
            unfreeze_last_n_layers=1, dropout=.1, training_balance={})

    def test_four_class_snapshot_registry_reload_and_six_slot_contract(self):
        artifact, optimization = neural.fit(self.items, self.targets, self.config)
        self.assertEqual(artifact["probability_model"], MODE)
        self.assertEqual(artifact["training"]["probability_model"], MODE)
        self.assertEqual(optimization["probability_model"], MODE)
        self.assertEqual(neural._RUNTIME[1].config.num_labels, 4)
        coarse_parameters = optimization["total_parameters"]
        warm = neural.predict(artifact, self.items)
        self.assertEqual(warm.shape, (len(self.items), 6))
        np.testing.assert_array_equal(warm[:, [1, 3]], np.zeros((len(self.items), 2)))
        np.testing.assert_allclose(warm.sum(-1), np.ones(len(self.items)), atol=1e-6)
        payload, checksum = serialize(artifact)
        registry = Registry(self.root / "registry.sqlite3")
        registry.save({"models": {}, "jobs": {}}, ("coarse", 1, checksum, payload))
        restored = registry.load("coarse", 1)
        neural._RUNTIME = None
        np.testing.assert_allclose(neural.predict(restored, self.items), warm, rtol=0, atol=1e-6)
        lora = {"rank": 2, "alpha": 4, "dropout": 0.0}
        flat_model, _, _ = neural._load(neural.prepare(self.config), lora)
        self.assertLess(coarse_parameters, sum(p.numel() for p in flat_model.parameters()))
        changed = copy.deepcopy(artifact)
        changed["probability_model"] = "flat"
        self.assertNotEqual(neural._identity(artifact), neural._identity(changed))
        settings = copy.deepcopy(self.config)
        settings["student_encoder"]["probability_model"] = "flat"
        with self.assertRaisesRegex(ValueError, "train from scratch"):
            neural.fit(self.items, self.targets, settings, prior=artifact)


if __name__ == "__main__":
    unittest.main()
