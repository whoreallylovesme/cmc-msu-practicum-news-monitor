"""Pure CPU loss/gradient checks; no encoder construction or weight downloads."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "runtime/ml"), str(ROOT)]
import neural


class PartialLossTests(unittest.TestCase):
    def test_zero_f_mass_matches_the_legacy_three_group_loss_and_gradient(self):
        logits = torch.tensor([[1., 2., 0., -2., -3., -4.], [-1., 0., 2., 1., .5, -.5]], requires_grad=True)
        expected = torch.tensor([[0., .6, .1, .3, 0., 0.], [0., .2, .6, .2, 0., 0.]])
        for temperature in (1., 2.):
            with self.subTest(temperature=temperature):
                actual = neural._loss(logits, expected, torch.tensor([True, True]), temperature, torch)
                logp = torch.log_softmax(logits / temperature, dim=-1)
                reference = expected.pow(1 / temperature)
                reference = reference / reference.sum(dim=-1, keepdim=True)
                coarse_logp = torch.stack([torch.logsumexp(logp[:, :2], -1), logp[:, 2],
                                          torch.logsumexp(logp[:, 3:5], -1)], -1)
                coarse_target = torch.stack([reference[:, :2].sum(-1), reference[:, 2], reference[:, 3:5].sum(-1)], -1)
                legacy = torch.nn.functional.kl_div(coarse_logp, coarse_target, reduction="none").sum(-1).mean() * temperature ** 2
                torch.testing.assert_close(actual, legacy)
                actual_gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
                legacy_gradient, = torch.autograd.grad(legacy, logits, retain_graph=True)
                torch.testing.assert_close(actual_gradient, legacy_gradient)

    def test_mixed_f_mass_has_the_four_group_gradient_without_forced_intensity(self):
        probabilities = torch.tensor([[.15, .15, .2, .1, .1, .3]])
        logits = probabilities.log().requires_grad_()
        target = torch.tensor([[0., .2, .3, .1, 0., .4]])
        loss = neural._loss(logits, target, torch.tensor([True]), 1., torch)
        gradient, = torch.autograd.grad(loss, logits)
        torch.testing.assert_close(gradient, torch.tensor([[.05, .05, -.1, .05, .05, -.1]]))
        for probability in ([.07, .13, .3, .02, .08, .4], [.19, .01, .3, .09, .01, .4]):
            matching = torch.tensor([probability]).log()
            marginal = neural._loss(matching, target, torch.tensor([True]), 1., torch)
            self.assertLess(abs(float(marginal)), 1e-6)

    def test_confidence_weights_scale_the_mean_over_examples_and_gradients(self):
        logits = torch.tensor([[1., 2., 0., -2., -3., -4.], [.2, -.4, .5, 0., .3, -.2],
                               [-.4, .1, .2, -.5, .7, 0.]], requires_grad=True)
        target = torch.tensor([[0., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 1.], [0., 1., 0., 0., 0., 0.]])
        partial = torch.tensor([False, True, True])
        weights = torch.tensor([1., .25, .5])
        logp = torch.log_softmax(logits, -1)
        per_example = torch.stack([-logp[0, 1], -logp[1, 5], -torch.logsumexp(logp[2, :2], -1)])
        reference = (per_example * weights).mean()
        actual = neural._loss(logits, target, partial, 1., torch, weights)
        torch.testing.assert_close(actual, reference)
        actual_gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
        reference_gradient, = torch.autograd.grad(reference, logits, retain_graph=True)
        torch.testing.assert_close(actual_gradient, reference_gradient)
        low_confidence = neural._loss(logits, target, partial, 1., torch, torch.full((3,), .1))
        ordinary = neural._loss(logits, target, partial, 1., torch)
        torch.testing.assert_close(low_confidence, ordinary * .1)

    def test_uneven_minibatch_accumulation_matches_the_weighted_full_group(self):
        logits = torch.tensor([[.2, -.1, .8, .4, -.2, .1], [0., .3, -.4, .2, .5, -.1],
                               [.1, .2, .3, .4, .5, .6], [.5, .4, .3, .2, .1, 0.],
                               [-.1, -.2, .2, -.3, .1, .4]], requires_grad=True)
        target = torch.eye(6)[torch.tensor([1, 2, 3, 5, 1])]
        partial = torch.tensor([True, False, True, True, False])
        weights = torch.tensor([1., .2, .3, .4, .8])
        reference = neural._loss(logits, target, partial, 1., torch, weights)
        accumulated = sum(neural._loss(logits[start:start + 2], target[start:start + 2],
                                      partial[start:start + 2], 1., torch, weights[start:start + 2])
                          * len(logits[start:start + 2]) / len(logits) for start in range(0, len(logits), 2))
        torch.testing.assert_close(accumulated, reference)
        actual_gradient, = torch.autograd.grad(accumulated, logits, retain_graph=True)
        reference_gradient, = torch.autograd.grad(reference, logits)
        torch.testing.assert_close(actual_gradient, reference_gradient)

    def test_known_mass_is_conditioned_and_legacy_flags_and_defaults_remain_valid(self):
        votes = np.asarray([.2002, .1001, .2002, .1001, .4004], dtype=np.float64)
        votes /= votes.sum()
        items = [{"partial_labels": True, "sample_weight": 1 - votes[0]}, {"polarity_only": True}, {}]
        targets = [[0., votes[1], votes[2], votes[3], 0., votes[4]], [0., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 1.]]
        values, partial, weights = neural._training_targets(items, targets)
        np.testing.assert_allclose(values[0], [0., .125, .25, .125, 0., .5])
        np.testing.assert_array_equal(partial, [True, True, False])
        np.testing.assert_allclose(weights, [.8, 1., 1.])

    def test_unknown_only_rows_and_invalid_weights_are_rejected_before_model_load(self):
        with self.assertRaises(ValueError):
            neural._training_targets([{"partial_labels": True, "sample_weight": 0.}], [[0.] * 6])
        for value in (0., -1., 1.01, 1.000000001, float("nan"), float("inf"), None):
            with self.subTest(weight=value), self.assertRaises(ValueError):
                neural._training_targets([{"sample_weight": value}], [[0., 0., 0., 0., 0., 1.]])

    def test_supervision_flags_and_weights_do_not_enter_model_inputs(self):
        item = {"text": "The weather is pleasant.", "entity": "weather conditions", "parent_text": "", "story_title": ""}
        baseline = neural._examples([item], "ChatGPT", ["Chat GPT"])
        tagged = {**item, "partial_labels": True, "polarity_only": True, "sample_weight": .2}
        self.assertEqual(neural._examples([tagged], "ChatGPT", ["Chat GPT"]), baseline)
