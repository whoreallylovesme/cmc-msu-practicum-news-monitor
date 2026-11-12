"""Check optimizer ownership and effective learning rates with tiny CPU tensors."""

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "ml"))
sys.path.insert(0, str(ROOT))

import neural


class StudentOptimizerTests(unittest.TestCase):
    @staticmethod
    def parameters():
        # Use actual PEFT parameter paths, with classifier and head interleaved.
        names = [
            "base_model.model.model.layers.0.attn.Wqkv.lora_A.default.weight",
            "base_model.model.head.modules_to_save.default.norm.weight",
            "base_model.model.classifier.modules_to_save.default.weight",
            "base_model.model.head.modules_to_save.default.dense.weight",
            "base_model.model.classifier.modules_to_save.default.bias",
            "base_model.model.model.layers.0.attn.Wo.lora_B.default.weight",
        ]
        return [(name, torch.nn.Parameter(torch.ones(2, dtype=torch.float64))) for name in names]

    def test_default_groups_keep_every_parameter_once_and_preserve_old_rates(self):
        named = self.parameters()
        groups = neural._optimizer_groups(named, 1e-4, 3e-4)
        self.assertEqual(len(groups), 2)
        actual_ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertCountEqual(actual_ids, [id(parameter) for _, parameter in named])
        self.assertEqual(len(actual_ids), len(set(actual_ids)))
        self.assertEqual([id(parameter) for parameter in groups[0]["params"]],
                         [id(parameter) for name, parameter in named if ".lora_" in name])
        self.assertEqual([id(parameter) for parameter in groups[1]["params"]],
                         [id(parameter) for name, parameter in named if ".lora_" not in name])
        self.assertEqual([group["lr"] for group in groups], [1e-4, 3e-4])

    def test_one_adamw_step_uses_lower_rate_only_for_pretrained_head(self):
        named = self.parameters()
        groups = neural._optimizer_groups(named, 1e-4, 3e-4, 3e-5)
        self.assertEqual(len(groups), 3)
        before = {name: parameter.detach().clone() for name, parameter in named}
        epsilon = 1e-12
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0, eps=epsilon)
        sum(parameter.sum() for _, parameter in named).backward()
        optimizer.step()
        self.assertEqual(len(optimizer.state), len(named))
        expected_rates = (1e-4, 3e-5, 3e-4, 3e-5, 3e-4, 1e-4)
        for (name, parameter), expected_rate in zip(named, expected_rates, strict=True):
            # Unit gradients make the first AdamW update equal to lr/(1+eps).
            torch.testing.assert_close(before[name] - parameter,
                                       torch.full_like(parameter, expected_rate / (1 + epsilon)),
                                       rtol=1e-8, atol=1e-12)
