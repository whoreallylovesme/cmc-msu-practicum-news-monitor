"""Persist actual encoder updates without embedding full weights in Registry."""

import copy
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open
from transformers import ModernBertConfig, ModernBertForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime" / "ml"), str(ROOT / "ut")]

import neural
import student
import test_student as fixtures
from lib.model_assets import file_sha256
from registry import Registry, serialize


class EncoderFinetuningTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("MONITOR_DATA_DIR", None)
        os.environ.pop("ML_ENCODER_CACHE_DIR", None)
        self.fixture = fixtures.StudentTests("test_training_reload_and_incremental_snapshots")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = self.fixture.config
        self.config["data_dir"] = str(self.fixture.root / "data")
        self.config["student_encoder"].update(
            unfreeze_last_n_layers=1, encoder_learning_rate=0.002,
            max_steps=1, gradient_accumulation=1,
        )
        torch.manual_seed(17)
        source = ModernBertForSequenceClassification(ModernBertConfig.from_pretrained(self.fixture.source, num_labels=2))
        source.save_pretrained(self.fixture.source)

    def fit(self, prior=None):
        return student.fit(self.fixture.items, self.fixture.targets, "encoder", self.config, prior=prior)

    def asset_path(self, artifact):
        return Path(self.config["data_dir"]) / artifact["encoder_state"]["asset"]["path"]

    def test_last_block_updates_without_growth_and_round_trips_registry(self):
        lora = {"rank": 2, "alpha": 4, "dropout": 0.0}
        initial, _, _ = neural._load(neural.prepare(self.config), lora)
        before = {name: parameter.detach().clone() for name, parameter in neural._encoder_parameters(initial).items()}
        total = sum(parameter.numel() for parameter in initial.parameters())
        artifact, metrics = self.fit()
        live = neural._RUNTIME[1]
        self.assertEqual(metrics["total_parameters"], total)
        self.assertFalse(metrics["base_encoder_frozen"])
        self.assertEqual(metrics["unfrozen_encoder_layers"], [1])
        self.assertEqual(artifact["encoder_state"]["layers"], [1])
        self.assertEqual(artifact["format_version"], 3)
        self.assertFalse(Path(artifact["encoder_state"]["asset"]["path"]).is_absolute())
        self.assertEqual(metrics["encoder_override_bytes"], self.asset_path(artifact).stat().st_size)
        self.assertTrue(any(not torch.equal(parameter, before[name])
                            for name, parameter in neural._encoder_parameters(live).items() if neural._encoder_layer(name) == 1))
        for name, parameter in neural._encoder_parameters(live).items():
            if neural._encoder_layer(name) == 0:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertTrue(all(not parameter.requires_grad for name, parameter in live.named_parameters()
                            if ".original_module." in name or ".embeddings." in name))
        named = [(name, parameter) for name, parameter in live.named_parameters() if parameter.requires_grad]
        groups = neural._optimizer_groups(named, 1e-4, 3e-4, 3e-5, 1e-5)
        rates = {id(parameter): group["lr"] for group in groups for parameter in group["params"]}
        self.assertEqual(sum(len(group["params"]) for group in groups), len(named))
        self.assertEqual(len(rates), len(named))
        self.assertTrue(all(rates[id(parameter)] == 1e-5 for parameter in neural._encoder_parameters(live).values()
                            if parameter.requires_grad))
        expected = student.predict(artifact, self.fixture.items)
        payload, checksum = serialize(artifact)
        self.assertLess(len(payload), artifact["encoder_state"]["asset"]["size_bytes"])
        registry = Registry(self.fixture.root / "registry.sqlite3")
        registry.save({"models": {}, "jobs": {}}, ("tiny", 1, checksum, payload))
        restored = registry.load("tiny", 1)
        neural._RUNTIME = None
        np.testing.assert_allclose(student.predict(restored, self.fixture.items), expected, rtol=0, atol=1e-6)

    def test_all_blocks_survive_incremental_training_with_frozen_encoder(self):
        self.config["student_encoder"]["unfreeze_last_n_layers"] = 2
        artifact, metrics = self.fit()
        self.assertEqual(artifact["encoder_state"]["layers"], [0, 1])
        self.assertEqual(metrics["unfrozen_encoder_layers"], [0, 1])
        with safe_open(self.asset_path(artifact), framework="pt") as tensors:
            original = {name: tensors.get_tensor(name).clone() for name in tensors.keys()}
        old_adapter = {name: values.copy() for name, values in artifact["state"].items()}
        self.config["student_encoder"]["unfreeze_last_n_layers"] = 0
        updated, incremental = self.fit(prior=artifact)
        self.assertTrue(incremental["warm_started"])
        self.assertTrue(incremental["base_encoder_frozen"])
        self.assertEqual(incremental["encoder_trainable_parameters"], 0)
        self.assertEqual(updated["encoder_state"], artifact["encoder_state"])
        for name, values in old_adapter.items():
            np.testing.assert_array_equal(artifact["state"][name], values)
        neural._RUNTIME = None
        student.predict(updated, self.fixture.items)
        actual = neural._encoder_parameters(neural._RUNTIME[1])
        for name, tensor in original.items():
            torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)

    def test_cold_load_verifies_asset_once_and_rejects_tampering_or_wrong_scope(self):
        artifact, _ = self.fit()
        path = self.asset_path(artifact)
        neural._RUNTIME = None
        with patch("lib.model_assets.file_sha256", wraps=file_sha256) as checksum:
            student.predict(artifact, self.fixture.items)
            student.predict(artifact, self.fixture.items)
        self.assertEqual(sum(Path(call.args[0]).resolve() == path.resolve() for call in checksum.call_args_list), 1)
        wrong = copy.deepcopy(artifact)
        wrong["encoder_state"]["keys"].pop()
        neural._RUNTIME = None
        with self.assertRaisesRegex(ValueError, "exactly"):
            student.predict(wrong, self.fixture.items)
        data = path.read_bytes()
        path.chmod(0o644)
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        neural._RUNTIME = None
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            student.predict(artifact, self.fixture.items)

    def test_relative_asset_relocates_with_monitor_data_dir(self):
        artifact, _ = self.fit()
        expected = student.predict(artifact, self.fixture.items)
        source = self.asset_path(artifact)
        relocated = self.fixture.root / "relocated"
        destination = relocated / artifact["encoder_state"]["asset"]["path"]
        destination.parent.mkdir(parents=True)
        shutil.copy2(source, destination)
        source.unlink()
        neural._RUNTIME = None
        with patch.dict(os.environ, {"MONITOR_DATA_DIR": str(relocated)}):
            np.testing.assert_allclose(student.predict(artifact, self.fixture.items), expected, rtol=0, atol=1e-6)

    def test_epoch_encoder_assets_remain_immutable_after_later_updates(self):
        self.config["student_encoder"].update(epochs=2, max_steps=None)
        checkpoints = []

        def checkpoint(artifact, _metrics):
            checkpoints.append((artifact, student.predict(artifact, self.fixture.items), self.asset_path(artifact).read_bytes()))

        student.fit(self.fixture.items, self.fixture.targets, "encoder", self.config, on_epoch=checkpoint)
        self.assertEqual(len(checkpoints), 2)
        self.assertNotEqual(checkpoints[0][0]["encoder_state"]["asset"]["sha256"],
                            checkpoints[1][0]["encoder_state"]["asset"]["sha256"])
        for artifact, expected, original_bytes in checkpoints:
            self.assertEqual(self.asset_path(artifact).read_bytes(), original_bytes)
            neural._RUNTIME = None
            np.testing.assert_allclose(student.predict(artifact, self.fixture.items), expected, rtol=0, atol=1e-6)

    def test_bfloat16_autocast_keeps_parameters_and_adam_moments_float32(self):
        self.config["student_encoder"].update(amp_dtype="bfloat16", gradient_checkpointing=True)
        optimizers = []
        original = torch.optim.AdamW

        def record(*args, **kwargs):
            optimizer = original(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        with patch("torch.optim.AdamW", side_effect=record):
            artifact, metrics = self.fit()
        self.assertEqual(metrics["amp_dtype"], "bfloat16")
        self.assertTrue(artifact["training"]["gradient_checkpointing"])
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in neural._RUNTIME[1].parameters()))
        moments = [value for state in optimizers[0].state.values() for name, value in state.items()
                   if name in {"exp_avg", "exp_avg_sq"}]
        self.assertTrue(moments)
        self.assertTrue(all(value.dtype == torch.float32 for value in moments))
        with safe_open(self.asset_path(artifact), framework="pt") as tensors:
            self.assertTrue(all(tensors.get_tensor(name).dtype == torch.float32 for name in tensors.keys()))

    def test_zero_scope_keeps_format_two_and_creates_no_encoder_assets(self):
        self.config["student_encoder"]["unfreeze_last_n_layers"] = 0
        artifact, metrics = self.fit()
        self.assertEqual(artifact["format_version"], 2)
        self.assertNotIn("encoder_state", artifact)
        self.assertTrue(metrics["base_encoder_frozen"])
        self.assertEqual(metrics["encoder_override_bytes"], 0)
        self.assertFalse((Path(self.config["data_dir"]) / "encoder-checkpoints").exists())

    def test_head_only_training_disables_lora(self):
        self.config["student_encoder"].update(unfreeze_last_n_layers=0, train_lora=False)
        _, metrics = self.fit()
        live = neural._RUNTIME[1]
        lora = [(name, parameter) for name, parameter in live.named_parameters() if ".lora_" in name]
        self.assertTrue(lora)
        self.assertTrue(all(not parameter.requires_grad for _, parameter in lora))
        head = [(name, parameter) for name, parameter in live.named_parameters()
                if ".modules_to_save." in name]
        self.assertTrue(head)
        self.assertTrue(all(parameter.requires_grad for _, parameter in head))
        self.assertEqual(metrics["trainable_parameters"], sum(parameter.numel() for _, parameter in head))
        self.assertTrue(metrics["base_encoder_frozen"])
        self.assertEqual(metrics["encoder_trainable_parameters"], 0)
        self.config["student_encoder"]["train_lora"] = "invalid"
        with self.assertRaisesRegex(ValueError, "train_lora"):
            self.fit()
