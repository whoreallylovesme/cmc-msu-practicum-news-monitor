import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "ml"))
sys.path.insert(0, str(ROOT))

from backends import causal_lm_ensemble
import student


class CausalLmEnsembleTests(unittest.TestCase):
    def test_evidence_prompt_is_loaded_from_external_file(self):
        path = Path(causal_lm_ensemble.__file__).parent.parent / "prompts" / "evidence_system.txt"
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_text(encoding="utf-8").strip(), causal_lm_ensemble.EVIDENCE_PROMPT)
        self.assertIn("EVIDENCE: <one short sentence>", causal_lm_ensemble.EVIDENCE_PROMPT)

    def test_label_only_prompt_is_a_separate_external_file(self):
        path = Path(causal_lm_ensemble.__file__).parent.parent / "prompts" / "label_only_system.txt"
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_text(encoding="utf-8").strip(), causal_lm_ensemble.LABEL_ONLY_PROMPT)
        self.assertIn("Return exactly one letter A, B, C, D, E or F", causal_lm_ensemble.LABEL_ONLY_PROMPT)
        self.assertNotIn("EVIDENCE:", causal_lm_ensemble.LABEL_ONLY_PROMPT)

    def test_parse_label_uses_final_class(self):
        self.assertEqual(causal_lm_ensemble.parse_label("EVIDENCE: Useful.\nCLASS: D"), 3)
        self.assertEqual(causal_lm_ensemble.parse_label("CLASS: A\nCLASS: F"), 5)
        with self.assertRaisesRegex(ValueError, "no CLASS"):
            causal_lm_ensemble.parse_label("EVIDENCE: Missing label")

    def test_two_irrelevant_votes_override_only_neutral(self):
        base = np.asarray([2, 2, 1, 3, 5])
        auxiliary = np.asarray([
            [5, 5, 5, 5, 5],
            [5, 1, 5, 1, 5],
            [1, 5, 1, 5, 1],
            [1, 1, 1, 5, 5],
        ])
        actual = causal_lm_ensemble.combine_labels(base, auxiliary)
        np.testing.assert_array_equal(actual, [5, 5, 1, 3, 5])

    def test_vote_does_not_use_identifiers_or_change_other_classes(self):
        auxiliary = np.zeros((4, 3), dtype=np.int64)
        auxiliary[0, 0] = 5
        actual = causal_lm_ensemble.combine_labels([0, 2, 4], auxiliary)
        np.testing.assert_array_equal(actual, [0, 2, 4])

    def test_invalid_vote_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "four auxiliary"):
            causal_lm_ensemble.combine_labels([2], np.zeros((3, 1), dtype=np.int64))

    def test_legacy_adapters_have_explicit_modes_and_stable_order(self):
        artifact = {"adapters": {
            name: {"path": f"/models/{name}", "sha256": "digest"}
            for name in ("epoch1_one_letter", "epoch2_one_letter", "epoch1_evidence", "epoch2_evidence")
        }}
        members = causal_lm_ensemble.ensemble_members(artifact)
        self.assertEqual(
            [(member["name"], member["mode"], member["path"]) for member in members],
            [
                ("epoch1_one_letter", "label_only", "/models/epoch1_one_letter"),
                ("epoch2_one_letter", "label_only", "/models/epoch2_one_letter"),
                ("epoch1_evidence", "evidence", "/models/epoch1_evidence"),
                ("epoch2_evidence", "evidence", "/models/epoch2_evidence"),
            ],
        )

    def test_generic_members_require_an_explicit_supported_mode(self):
        artifact = {"members": [
            {"name": str(index), "path": f"/models/{index}", "sha256": "digest"}
            for index in range(4)
        ]}
        with self.assertRaisesRegex(ValueError, "mode"):
            causal_lm_ensemble.ensemble_members(artifact)
        artifact["members"][0]["mode"] = "unknown"
        with self.assertRaisesRegex(ValueError, "mode"):
            causal_lm_ensemble.ensemble_members(artifact)

    def test_artifact_verification_accepts_generic_members_and_legacy_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            adapters = {}
            for name, _ in causal_lm_ensemble.LEGACY_ADAPTER_MODES:
                path = root / name
                path.mkdir()
                (path / "adapter.bin").write_bytes(name.encode("utf-8"))
                adapters[name] = {
                    "path": str(path),
                    "sha256": causal_lm_ensemble.directory_sha256(path),
                }
            legacy = {
                "architecture": causal_lm_ensemble.LEGACY_PROTOCOL,
                "assets": str(assets),
                "base_assets": {},
                "adapters": adapters,
            }
            causal_lm_ensemble.verify_artifact(legacy)
            generic = {
                "architecture": causal_lm_ensemble.PROTOCOL,
                "assets": str(assets),
                "base_assets": {},
                "members": [
                    {**adapters[name], "name": name, "mode": mode}
                    for name, mode in causal_lm_ensemble.LEGACY_ADAPTER_MODES
                ],
            }
            causal_lm_ensemble.verify_artifact(generic)
            legacy["adapters"] = {
                **adapters,
                "extra": {"path": str(root / "missing"), "sha256": "0" * 64},
            }
            with self.assertRaisesRegex(ValueError, "Adapter checksum mismatch extra"):
                causal_lm_ensemble.verify_artifact(legacy)

    def test_predict_uses_member_modes_instead_of_member_names(self):
        artifact = {
            "architecture": causal_lm_ensemble.PROTOCOL,
            "assets": "/models/base",
            "members": [
                {"name": "epoch1_evidence", "path": "/models/labels", "mode": "label_only"},
                {"name": "epoch1_one_letter", "path": "/models/evidence", "mode": "evidence"},
                {"name": "third", "path": "/models/third", "mode": "label_only"},
                {"name": "fourth", "path": "/models/fourth", "mode": "evidence"},
            ],
        }
        item = {"id": "one", "story_id": "story", "text": "Text"}
        with (
            patch.object(causal_lm_ensemble, "verify_artifact"),
            patch.object(causal_lm_ensemble, "resolve_device", return_value="cpu"),
            patch.object(causal_lm_ensemble, "load_tokenizer", return_value="tokenizer"),
            patch.object(causal_lm_ensemble, "load_model", side_effect=["base", "first", "second", "third", "fourth"]) as load_model,
            patch.object(causal_lm_ensemble, "generated_labels", side_effect=[[1], [5], [5]]) as generated,
            patch.object(causal_lm_ensemble, "logit_labels", side_effect=[[1], [1]]) as logits,
            patch.object(causal_lm_ensemble, "release"),
        ):
            actual = causal_lm_ensemble.predict(artifact, [item])
        self.assertEqual(load_model.call_args_list, [
            call(artifact, "tokenizer"),
            call(artifact, "tokenizer", "/models/labels"),
            call(artifact, "tokenizer", "/models/evidence"),
            call(artifact, "tokenizer", "/models/third"),
            call(artifact, "tokenizer", "/models/fourth"),
        ])
        self.assertEqual([args.args[0] for args in generated.call_args_list], ["base", "second", "fourth"])
        self.assertEqual([args.args[0] for args in logits.call_args_list], ["first", "third"])
        np.testing.assert_array_equal(actual, np.eye(6)[[1]])

    def test_class_tokens_must_be_single_and_distinct(self):
        class SingleTokenTokenizer:
            def encode(self, letter, add_special_tokens=False):
                return [ord(letter)]

        self.assertEqual(causal_lm_ensemble.class_token_ids(SingleTokenTokenizer()), [65, 66, 67, 68, 69, 70])

        class SplitTokenTokenizer:
            def encode(self, letter, add_special_tokens=False):
                return [1, 2] if letter == "A" else [ord(letter)]

        with self.assertRaisesRegex(ValueError, "exactly one token"):
            causal_lm_ensemble.class_token_ids(SplitTokenTokenizer())

        class DuplicateTokenTokenizer(SingleTokenTokenizer):
            def encode(self, letter, add_special_tokens=False):
                return [1]

        with self.assertRaisesRegex(ValueError, "distinct"):
            causal_lm_ensemble.class_token_ids(DuplicateTokenTokenizer())

    def test_student_static_registry_dispatches_generic_and_legacy_artifacts(self):
        self.assertIs(student.BACKENDS[causal_lm_ensemble.PROTOCOL], causal_lm_ensemble)
        self.assertIs(student.BACKENDS[causal_lm_ensemble.LEGACY_PROTOCOL], causal_lm_ensemble)
        expected = np.eye(6)[[5]]
        item = {"id": "one", "story_id": "story", "text": "Unrelated"}
        for protocol in (causal_lm_ensemble.PROTOCOL, causal_lm_ensemble.LEGACY_PROTOCOL):
            with self.subTest(protocol=protocol):
                artifact = {"kind": "encoder", "architecture": protocol}
                with patch("backends.causal_lm_ensemble.predict", return_value=expected) as predict:
                    actual = student.predict(artifact, [item])
                predict.assert_called_once_with(artifact, [item])
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
