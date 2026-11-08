"""Exercise adapter snapshots with a tiny local ModernBERT, without downloads."""

import copy
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import ModernBertConfig, ModernBertForMaskedLM, ModernBertForSequenceClassification, PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "ml"))
sys.path.insert(0, str(ROOT))

import neural
import student


class StudentLossTests(unittest.TestCase):
    def test_temperature_preserves_zero_mass_for_partial_polarity(self):
        logits = torch.tensor([[1.0, 2.0, 0.0, -2.0, -3.0, -4.0]], dtype=torch.float64, requires_grad=True)
        expected = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        temperature = 10.0
        loss = neural._loss(logits, expected, torch.tensor([True]), temperature, torch)
        # Match fp32 log probabilities and their promotion by the float64 targets.
        reference = -torch.logsumexp(
            torch.log_softmax(logits.float() / temperature, dim=-1)[0, :2], dim=-1,
        ).to(expected.dtype) * temperature ** 2
        torch.testing.assert_close(loss, reference)
        actual_gradient, = torch.autograd.grad(loss, logits, retain_graph=True)
        reference_gradient, = torch.autograd.grad(reference, logits)
        torch.testing.assert_close(actual_gradient, reference_gradient)
        torch.testing.assert_close(expected, torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64))


class StudentBalanceTests(unittest.TestCase):
    def test_default_and_invalid_settings(self):
        items, targets, partial = [{"id": "one"}], np.eye(6)[:1], np.asarray([False])
        for settings in ({}, {"training_balance": None}):
            weights, metadata = neural._balance_weights(items, targets, partial, settings)
            np.testing.assert_array_equal(weights, np.ones(1, dtype=np.float32))
            self.assertIsNone(metadata)
        for spec in (False, {"exponent": float("nan")}, {"exponent": -1},
                     {"max_class_weight": .5}, {"source_weights": {"primary": 0}}):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                neural._balance_weights(items, targets, partial, {"training_balance": spec})
        for value in (-.1, 1, True, float("nan"), "0.1"):
            with self.subTest(dropout=value), self.assertRaises(ValueError):
                neural._dropout_overrides(value)
        self.assertEqual(neural._dropout_overrides(None), {})

    def test_partial_class_mass_has_inverse_sqrt_ratio_and_keeps_confidence_separate(self):
        items = [{"id": str(i), "source": "weather", "sample_weight": .01 if i == 9 else 1}
                 for i in range(10)]
        targets = np.asarray([[.25, .75, 0, 0, 0, 0]] * 9 + [[0, 0, 0, 0, 0, 1]])
        before = targets.copy()
        weights, metadata = neural._balance_weights(items, targets, np.ones(10, dtype=bool),
            {"training_balance": {"exponent": .5, "max_class_weight": 4}})
        self.assertAlmostEqual(float(weights[-1] / weights[0]), 3, places=6)
        self.assertAlmostEqual(float(weights.mean()), 1, places=6)
        self.assertEqual(metadata["groups"][0]["class_mass"], [9, 0, 0, 1])
        np.testing.assert_array_equal(targets, before)
        changed = copy.deepcopy(items)
        changed[-1]["sample_weight"] = 1
        np.testing.assert_array_equal(neural._balance_weights(changed, targets, np.ones(10, dtype=bool),
            {"training_balance": {}})[0], weights)

    def test_soft_example_factor_is_linear_and_sources_normalize_independently(self):
        items = [{"id": str(i), "training_source": "weather", "source": "ignored"} for i in range(5)]
        targets = np.asarray([[0, 1, 0, 0, 0, 0]] * 3 +
                             [[0, 0, 0, 0, 0, 1], [0, .5, 0, 0, 0, .5]])
        items.append({"id": "air", "source": "airline"})
        targets = np.vstack((targets, np.eye(6)[2]))
        weights, metadata = neural._balance_weights(items, targets, np.ones(6, dtype=bool),
            {"training_balance": {"source_weights": {"weather": 2, "airline": .5}, "max_class_weight": 1.1}})
        self.assertAlmostEqual(float(weights[:5].mean()), 2, places=6)
        self.assertAlmostEqual(float(weights[-1]), .5, places=6)
        self.assertAlmostEqual(float(weights[4]), float((weights[0] + weights[3]) / 2), places=6)
        self.assertTrue(all(max(group["class_factors"]) <= 1.1 for group in metadata["groups"]))
        self.assertEqual(neural._examples(items, "ChatGPT", []), neural._examples(
            [{key: value for key, value in row.items() if key not in {"training_source", "source"}} for row in items],
            "ChatGPT", []))

    def test_partial_and_full_labels_have_separate_class_masses(self):
        targets = np.eye(6)[[0, 0, 1, 5, 0, 0, 1, 5]]
        partial = np.asarray([True] * 4 + [False] * 4)
        weights, metadata = neural._balance_weights([{"id": str(i)} for i in range(8)], targets, partial,
                                                   {"training_balance": {}})
        self.assertEqual(len(metadata["groups"]), 2)
        self.assertAlmostEqual(float(weights[0]), float(weights[2]), places=6)
        self.assertGreater(float(weights[6]), float(weights[4]))
        self.assertAlmostEqual(float(weights.mean()), 1, places=6)

    def test_balance_and_confidence_multiply_the_whole_example_gradient(self):
        logits = torch.tensor([[1., 0, -1, 0, 0, 0], [0, 1, 0, -1, 0, 0]], requires_grad=True)
        targets = torch.tensor([[0., 1, 0, 0, 0, 0], [0., 0, 0, 0, 0, 1]])
        partial = torch.tensor([True, False])
        confidence, balance = torch.tensor([.2, 1]), torch.tensor([3., .5])
        loss = neural._loss(logits, targets, partial, 1, torch, confidence, balance)
        individual = torch.stack([neural._loss(logits[i:i + 1], targets[i:i + 1], partial[i:i + 1], 1, torch)
                                  for i in range(2)])
        reference = (individual * confidence * balance).mean()
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(torch.autograd.grad(loss, logits, retain_graph=True)[0],
                                   torch.autograd.grad(reference, logits)[0])


class StudentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "base"
        self.source.mkdir()
        vocabulary = {
            token: index for index, token in enumerate([
                "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "Target", "entity", ":",
                "ChatGPT", "Chat", "GPT", "Comment", "Parent", "comment", "context", "only",
                "Story", "title", "Aliases", "good", "bad", "useful", "awful", "neutral",
                "news", "other", "weather", "is", "very", "I", "like", "hate",
            ])
        }
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])],
        )
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
            cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=64,
        ).save_pretrained(self.source)
        torch.manual_seed(7)
        model_config = ModernBertConfig(
            vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64,
            local_attention=16, global_attn_every_n_layers=1, reference_compile=False,
            pad_token_id=1, cls_token_id=2, sep_token_id=3, bos_token_id=2, eos_token_id=3,
            classifier_pooling="mean",
        )
        ModernBertForMaskedLM(model_config).save_pretrained(self.source)
        self.config = {
            "entity": "ChatGPT", "aliases": ["Chat GPT"], "score_statistic": "mean",
            "student_encoder": {
                "model_weights": {"path": str(self.source), "is_local": True},
                "cache_dir": str(self.root / "cache"), "device": "cpu", "dtype": "float32",
                "max_length": 48, "batch_size": 2, "gradient_accumulation": 2,
                "threads": 1, "epochs": 1, "incremental_epochs": 1,
                "learning_rate": 0.005, "head_learning_rate": 0.01,
                "lora_rank": 2, "lora_alpha": 4, "lora_dropout": 0.0, "seed": 42,
            },
        }
        texts = ["ChatGPT is very awful", "I hate ChatGPT", "ChatGPT news", "ChatGPT is useful", "ChatGPT is very good", "other weather news"]
        self.items = [
            {"id": str(index), "story_id": str(index % 2), "text": text,
             "story_title": "news", "parent_text": "ChatGPT", "published_at": "2026-01-01T00:00:00Z"}
            for index, text in enumerate(texts)
        ]
        self.targets = np.eye(6, dtype=np.float32)
        self.addCleanup(setattr, neural, "_RUNTIME", None)

    def test_training_reload_and_incremental_snapshots(self):
        self.items[0]["polarity_only"] = True
        artifact, metrics = student.fit(self.items, self.targets, "encoder", self.config)
        self.assertEqual(metrics["iterations"], 2)
        self.assertFalse(metrics["warm_started"])
        self.assertGreater(metrics["trainable_parameters"], 0)
        self.assertTrue(metrics["base_encoder_frozen"])
        self.assertLess(metrics["adapter_bytes"], 1024 * 1024)
        self.assertEqual(metrics["polarity_only_items"], 1)
        self.assertEqual(artifact["training"]["seed"], 42)
        self.assertEqual(artifact["training"]["gradient_accumulation"], 2)
        self.assertEqual(artifact["training"]["optimizer"]["learning_rate"], 0.005)
        self.assertFalse(artifact["training"]["optimizer"]["state_restored"])
        self.assertNotIn("mention_policy", artifact)
        self.assertNotIn("dropout", artifact["training"])
        self.assertNotIn("training_balance", artifact["training"])
        self.assertTrue(all(isinstance(value, np.ndarray) for value in artifact["state"].values()))
        before = student.predict(artifact, self.items)
        np.testing.assert_allclose(before.sum(axis=1), np.ones(len(self.items)), atol=1e-6)
        self.assertTrue(np.isfinite(before).all())

        payload = pickle.dumps(artifact, protocol=5)
        self.assertLess(len(payload), 1024 * 1024)
        restored = pickle.loads(payload)
        neural._RUNTIME = None
        np.testing.assert_allclose(student.predict(restored, self.items), before, atol=1e-6)

        old_state = {name: value.copy() for name, value in artifact["state"].items()}
        updated, incremental = student.fit(self.items, self.targets, "encoder", self.config, prior=artifact)
        self.assertTrue(incremental["warm_started"])
        for name, values in old_state.items():
            np.testing.assert_array_equal(artifact["state"][name], values)
        self.assertTrue(any(not np.array_equal(old_state[name], updated["state"][name]) for name in old_state))
        # Switching back to an old revision must restore its predictions exactly.
        np.testing.assert_allclose(student.predict(artifact, self.items), before, atol=1e-6)

    def test_partial_polarity_does_not_invent_an_intensity(self):
        logits = torch.tensor([[1.0, 2.0, 0.0, -2.0, -3.0, -4.0]], requires_grad=True)
        expected = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        loss = neural._loss(logits, expected, torch.tensor([True]), 1.0, torch)
        reference = -torch.log(torch.softmax(logits, dim=-1)[0, :2].sum())
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertLess(float(logits.grad[0, 1]), 0)
        self.assertGreater(float(logits.grad[0, 5]), 0)

    def test_two_class_base_preserves_encoder_and_head_with_six_class_output(self):
        lora = {"rank": 2, "alpha": 4, "dropout": 0.0}
        torch.manual_seed(42)
        mlm, _, _ = neural._load(neural.prepare(self.config), lora)
        initial_classifier = mlm.base_model.model.classifier.modules_to_save["default"].weight.detach().clone()
        source = ModernBertForSequenceClassification(ModernBertConfig.from_pretrained(self.source, num_labels=2))
        source.save_pretrained(self.source)
        torch.manual_seed(42)
        loaded, _, _ = neural._load(neural.prepare(self.config), lora)
        core = loaded.base_model.model
        self.assertEqual(core.classifier.modules_to_save["default"].out_features, 6)
        torch.testing.assert_close(core.classifier.modules_to_save["default"].weight, initial_classifier)
        for name, value in source.model.state_dict().items():
            adapted = name.replace(".attn.Wqkv.", ".attn.Wqkv.base_layer.").replace(".attn.Wo.", ".attn.Wo.base_layer.")
            torch.testing.assert_close(core.model.state_dict()[adapted], value)
        head = core.head.modules_to_save["default"].state_dict()
        for name, value in source.head.state_dict().items():
            torch.testing.assert_close(head[name], value)

    def test_training_dropout_keeps_parameters_and_eval_predictions_unchanged(self):
        bundle, lora = neural.prepare(self.config), {"rank": 2, "alpha": 4, "dropout": 0.0}
        torch.manual_seed(42)
        baseline, tokenizer, device = neural._load(bundle, lora)
        torch.manual_seed(42)
        regularized, _, _ = neural._load(bundle, lora, dropout=.2)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), sum(p.numel() for p in regularized.parameters()))
        self.assertTrue(all(p.dtype == torch.float32 for p in regularized.parameters()))
        for name in ("embedding_dropout", "attention_dropout", "mlp_dropout", "classifier_dropout"):
            self.assertEqual(getattr(regularized.config, name), .2)
            self.assertEqual(getattr(baseline.config, name), 0)
        for (name, value), (other_name, other) in zip(baseline.named_parameters(), regularized.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(value, other, rtol=0, atol=0)
        batch = neural._batch(tokenizer, self.items, "ChatGPT", ["Chat GPT"], bundle, device)
        baseline.eval()
        regularized.eval()
        with torch.inference_mode():
            torch.testing.assert_close(baseline(**batch).logits, regularized(**batch).logits, rtol=0, atol=0)

    def test_regularization_metadata_and_reload_keep_confidence_separate(self):
        self.config["student_encoder"].update(dropout=.2, max_steps=1, training_balance={})
        self.targets[0] = self.targets[1]
        self.items[0]["sample_weight"] = .4
        artifact, metrics = student.fit(self.items, self.targets, "encoder", self.config)
        self.assertEqual(artifact["training"]["dropout"], .2)
        self.assertEqual(metrics["training_balance"], artifact["training"]["training_balance"])
        balance = metrics["training_balance"]
        self.assertNotEqual(balance["weights"]["sha256"], balance["effective_weights"]["sha256"])
        self.assertAlmostEqual(balance["confidence_weights"]["sum"], 5.4, places=6)
        before = student.predict(artifact, self.items)
        neural._RUNTIME = None
        np.testing.assert_allclose(student.predict(artifact, self.items), before, atol=1e-6)

    def test_nonclassifier_checkpoint_shape_mismatch_is_rejected(self):
        from safetensors.torch import load_file, save_file

        source = ModernBertForSequenceClassification(ModernBertConfig.from_pretrained(self.source, num_labels=2))
        for name in ("head.dense.weight", "model.layers.0.attn.Wqkv.weight"):
            with self.subTest(parameter=name):
                source.save_pretrained(self.source)
                filename = self.source / "model.safetensors"
                weights = load_file(str(filename))
                weights[name] = weights[name][:1].contiguous()
                save_file(weights, str(filename), metadata={"format": "pt"})
                with self.assertRaisesRegex(ValueError, "encoder and head must match"):
                    neural._load(neural.prepare(self.config), {"rank": 2, "alpha": 4, "dropout": 0.0})

    def test_marginal_relevance_and_six_class_metrics_are_separate(self):
        distribution = [0.14, 0.14, 0.14, 0.14, 0.14, 0.3]
        score, relevant = student.score_distribution(distribution)
        self.assertTrue(relevant)
        self.assertIsNotNone(score)
        metrics = student.label_metrics([2], [distribution])
        self.assertEqual(metrics["relevance_accuracy"], 1)
        self.assertEqual(metrics["class_accuracy"], 0)

    def test_cache_override_applies_to_existing_artifacts(self):
        bundle = neural.prepare(self.config)
        with patch.dict("os.environ", {"ML_ENCODER_CACHE_DIR": str(self.root / "different-cache")}):
            path = neural._model_directory(bundle)
        self.assertEqual(path.parent, self.root / "different-cache")

    def test_changed_base_or_entity_cannot_warm_start(self):
        artifact, _ = student.fit(self.items, self.targets, "encoder", self.config)
        changed = copy.deepcopy(self.config)
        changed["entity"] = "Other entity"
        with self.assertRaisesRegex(ValueError, "train from scratch"):
            student.fit(self.items, self.targets, "encoder", changed, prior=artifact)
        changed_base = copy.deepcopy(self.config)
        changed_base["student_encoder"]["model_weights"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "train from scratch"):
            student.fit(self.items, self.targets, "encoder", changed_base, prior=artifact)
        with self.assertRaisesRegex(ValueError, "supported student kind"):
            student.fit(self.items, self.targets, "hard_label", self.config)

    def test_explicit_step_budget_is_recorded(self):
        self.config["student_encoder"].update(epochs=3, max_steps=1)
        artifact, metrics = student.fit(self.items, self.targets, "encoder", self.config)
        self.assertEqual(metrics["iterations"], 1)
        self.assertEqual(artifact["training"]["max_steps"], 1)
        self.assertEqual(artifact["training"]["epochs_requested"], 3)
        self.assertEqual(artifact["training"]["epochs_completed"], 0)

    def test_epoch_callbacks_can_validate_immutable_snapshots(self):
        self.config["student_encoder"]["epochs"] = 2
        checkpoints = []

        def save_epoch(artifact, metrics):
            checkpoints.append((artifact, student.predict(artifact, self.items), metrics))

        final, _ = student.fit(self.items, self.targets, "encoder", self.config, on_epoch=save_epoch)
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(checkpoints[0][2]["epoch"], 1)
        np.testing.assert_allclose(student.predict(checkpoints[0][0], self.items), checkpoints[0][1], atol=1e-6)
        np.testing.assert_allclose(student.predict(final, self.items), checkpoints[-1][1], atol=1e-6)
