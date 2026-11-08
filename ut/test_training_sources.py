"""Selected human CSVs remain pinned, ordered and separate from model features."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.supervision import load_auxiliary
from lib import training_sources


class TrainingSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        # Keep service-container cache overrides out of these isolated fixtures.
        os.environ.pop("MONITOR_DATA_DIR", None)
        self.root = Path(self.temporary.name)
        self.weather = self.root / "weather.csv"
        self.weather.write_text("id,tweet,s1,s2,s3,s4,s5,gold\n"
                                "1,Raw weather text,0.201,0.2,0.1,0.299,0.201,SECRET\n"
                                "2,Uncertain but useful,0.8,0.1,0.1,0,0,SECRET\n"
                                "3,Unknown only,1,0,0,0,0,SECRET\n")
        self.airline = self.root / "airline.csv"
        self.airline.write_text("tweet_id,airline,text,airline_sentiment,annotation_confidence\n"
                                "100,United,Raw airline text,positive,0.4\n")
        self.selection = self.root / "selection.yaml"
        self.sources = {
            "weather": {"format": "crowdflower_weather_csv", "asset": self.asset(self.weather),
                        "split": "train", "entity": "weather conditions",
                        "items": [{"id": "partly-sunny:2", "group_id": "group:2"},
                                  {"id": "partly-sunny:1", "group_id": "group:1"}]},
            "airline": {"format": "airline_csv", "asset": self.asset(self.airline), "split": "train",
                        "items": [{"id": "airline:100:0", "group_id": "airline-group"}]},
        }
        self.network = patch("requests.get", side_effect=AssertionError("Offline CSV tests cannot use the network"))
        self.network.start()
        self.addCleanup(self.network.stop)

    @staticmethod
    def asset(path):
        return {"path": str(path), "is_local": True, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size}

    def configured(self):
        self.selection.write_text(yaml.safe_dump({"version": 1, "sources": self.sources}, sort_keys=False))
        return {"data_dir": str(self.root / "cache"), "auxiliary_supervision": {
            "path": None, "incremental_replay_items": 2,
            "additional_sources": [{"name": name, "format": source["format"], "asset": source["asset"],
                                    "selection": self.asset(self.selection)} for name, source in self.sources.items()],
        }}

    def test_exact_selection_order_raw_text_and_partial_targets(self):
        items, targets, metadata = load_auxiliary(self.configured())
        self.assertEqual([item["id"] for item in items], ["aux:partly-sunny:2", "aux:partly-sunny:1", "aux:airline:100:0"])
        self.assertEqual(items[0]["text"], "Uncertain but useful")
        self.assertTrue(items[0]["partial_labels"])
        self.assertAlmostEqual(items[0]["sample_weight"], 0.2)
        self.assertEqual(targets[0], [0, .5, .5, 0, 0, 0])
        raw = [.201, .2, .1, .299, .201]
        normalized = [value / sum(raw) for value in raw]
        known = sum(normalized[1:])
        self.assertEqual(items[1]["sample_weight"], 1.0 - normalized[0])
        self.assertEqual(targets[1], [0, normalized[1] / known, normalized[2] / known,
                                    normalized[3] / known, 0, normalized[4] / known])
        self.assertEqual(items[2]["entity"], "United Airlines")
        self.assertEqual(items[2]["sample_weight"], 1)
        self.assertEqual(targets[2], [0, 0, 0, 1, 0, 0])
        self.assertEqual(metadata["available_items"], 3)
        self.assertTrue(all(set(item) == {"id", "story_id", "entity", "text", "parent_text", "story_title",
                                         "parent_id", "partial_labels", "sample_weight"} for item in items))
        self.assertTrue(all(item["parent_text"] == item["story_title"] == "" for item in items))

    def test_only_zero_known_mass_is_skipped(self):
        self.sources["weather"]["items"].append({"id": "partly-sunny:3", "group_id": "group:3"})
        items, _, metadata = load_auxiliary(self.configured())
        self.assertEqual(len(items), 3)
        self.assertEqual(metadata["sources"][0]["skipped_uninformative_ids"], ["partly-sunny:3"])

    def test_weather_class_uses_original_plurality_and_preserves_known_weight(self):
        self.weather.write_text(self.weather.read_text()
                                + "4,F plurality,0.1,0.2,0.2,0.14,0.36,SECRET\n")
        self.sources["weather"]["asset"] = self.asset(self.weather)
        self.sources["weather"]["items"] = [{"id": "partly-sunny:4", "group_id": "group:4"}]
        soft_items, soft_targets, _ = load_auxiliary(self.configured())
        self.assertAlmostEqual(soft_targets[0][5], .4)
        self.sources["weather"]["target_mode"] = "class"
        items, targets, metadata = load_auxiliary(self.configured())
        self.assertEqual(items, soft_items)
        self.assertEqual(targets[0], [0, 0, 0, 0, 0, 1])
        self.assertAlmostEqual(items[0]["sample_weight"], .9)
        self.assertEqual(metadata["sources"][0]["target_mode"], "class")
        self.assertEqual(metadata["sources"][0]["format_version"], "selected-human-csv-class-v1")

    def test_weather_class_skips_unknown_and_tied_original_winners(self):
        self.weather.write_text(self.weather.read_text()
                                + "5,Tied known votes,0,0.3,0.3,0.2,0.2,SECRET\n")
        self.sources["weather"]["asset"] = self.asset(self.weather)
        self.sources["weather"]["items"].extend([
            {"id": "partly-sunny:5", "group_id": "group:5"},
            {"id": "partly-sunny:3", "group_id": "group:3"},
        ])
        soft_items, _, soft_metadata = load_auxiliary(self.configured())
        self.assertEqual(len(soft_items), 4)
        self.assertEqual(soft_metadata["sources"][0]["skipped_uninformative_ids"], ["partly-sunny:3"])
        self.sources["weather"]["target_mode"] = "class"
        items, targets, metadata = load_auxiliary(self.configured())
        self.assertEqual([item["id"] for item in items], ["aux:partly-sunny:1", "aux:airline:100:0"])
        self.assertEqual(targets[0], [0, 0, 0, 1, 0, 0])
        self.assertEqual(metadata["sources"][0]["skipped_uninformative_ids"],
                         ["partly-sunny:2", "partly-sunny:5", "partly-sunny:3"])

    def test_weather_target_mode_is_validated(self):
        for target_mode in ("unknown", None, True):
            with self.subTest(target_mode=target_mode):
                self.sources["weather"]["target_mode"] = target_mode
                with self.assertRaisesRegex(ValueError, "target_mode"):
                    load_auxiliary(self.configured())

    def test_mixed_weather_retains_uncertainty_and_the_human_F_decision(self):
        self.weather.write_text(self.weather.read_text()
                                + "4,F plurality,0.1,0.2,0.2,0.14,0.36,SECRET\n")
        self.sources["weather"].update(
            asset=self.asset(self.weather), target_mode="mixed", class_mix=0.5,
            items=[{"id": "partly-sunny:4", "group_id": "group:4"}],
        )
        items, targets, metadata = load_auxiliary(self.configured())
        self.assertGreater(targets[0][5], 0.5)
        self.assertLess(targets[0][5], 1)
        self.assertTrue(all(targets[0][index] > 0 for index in (1, 2, 3)))
        self.assertEqual(targets[0][0], 0)
        self.assertEqual(targets[0][4], 0)
        self.assertAlmostEqual(sum(targets[0]), 1)
        self.assertAlmostEqual(items[0]["sample_weight"], 0.9)
        self.assertTrue(items[0]["partial_labels"])
        self.assertEqual(items[0]["text"], "F plurality")
        self.assertEqual(metadata["sources"][0]["class_mix"], 0.5)
        self.assertEqual(metadata["sources"][0]["format_version"], "selected-human-csv-mixed-v1")

    def test_mixed_weather_endpoints_and_invalid_coefficient(self):
        row = {"tweet": "Raw weather", "s1": "0.1", "s2": "0.1", "s3": "0.2", "s4": "0.5", "s5": "0.1"}
        selected = {"entity": "weather", "target_mode": "mixed"}
        soft = training_sources._weather(row, {**selected, "target_mode": "distribution"})
        hard = training_sources._weather(row, {**selected, "target_mode": "class"})
        self.assertEqual(training_sources._weather(row, {**selected, "class_mix": 0}), soft)
        self.assertEqual(training_sources._weather(row, {**selected, "class_mix": 1}), hard)
        for invalid in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "class_mix"):
                    training_sources._weather(row, {**selected, "class_mix": invalid})

    def test_balancing_source_tags_are_opt_in_and_leave_supervision_unchanged(self):
        config = self.configured()
        items, targets, metadata = load_auxiliary(config)
        self.assertTrue(all("training_source" not in item for item in items))
        config["student_encoder"] = {"training_balance": {}}
        tagged, tagged_targets, tagged_metadata = load_auxiliary(config)
        self.assertEqual([item["training_source"] for item in tagged], ["weather", "weather", "airline"])
        self.assertEqual([{key: value for key, value in item.items() if key != "training_source"}
                          for item in tagged], items)
        self.assertEqual(tagged_targets, targets)
        self.assertEqual(tagged_metadata, metadata)

    def test_default_metadata_is_unchanged_and_class_mode_changes_revision(self):
        config = self.configured()
        _, _, before = load_auxiliary(config)
        weather = before["sources"][0]
        self.assertEqual(weather["format_version"], "selected-human-csv-v1")
        self.assertNotIn("target_mode", weather)
        self.assertEqual(set(weather), {"name", "format", "format_version", "source_sha256",
                                       "source_size_bytes", "selection_sha256", "selection_size_bytes",
                                       "source_ids", "group_ids", "skipped_uninformative_ids", "items",
                                       "available_items"})
        original_selection = training_sources._selection

        def with_class_mode(path, name, format_name, source_sha):
            selected = original_selection(path, name, format_name, source_sha)
            return {**selected, "target_mode": "class"} if name == "weather" else selected

        with patch.object(training_sources, "_selection", side_effect=with_class_mode):
            _, _, after = load_auxiliary(config)
        # The format revision is in the existing identity allowlist, even when
        # descriptor bytes/SHA are identical in this controlled comparison.
        self.assertEqual(after["sources"][0]["selection_sha256"], weather["selection_sha256"])
        self.assertNotEqual(after["sha256"], before["sha256"])

    def test_nonfinite_or_negative_human_shares_are_rejected(self):
        for value in ("nan", "-0.1"):
            with self.subTest(value=value):
                original = self.weather.read_text()
                self.weather.write_text(original.replace("0.201,0.2,0.1,0.299,0.201", f"{value},0.2,0.1,0.299,0.201"))
                self.sources["weather"]["asset"] = self.asset(self.weather)
                with self.assertRaisesRegex(ValueError, "finite"):
                    load_auxiliary(self.configured())
                self.weather.write_text(original)
                self.sources["weather"]["asset"] = self.asset(self.weather)

    def test_training_weights_and_gold_never_change_encoded_features(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/ml"))
        import neural

        items, _, _ = load_auxiliary(self.configured())
        features = neural._examples(items, "configured entity", [])
        modified = copy.deepcopy(items)
        for item in modified:
            item.update(sample_weight=0.01, partial_labels=False, polarity_only=True,
                        gold_polarity="SECRET_GOLD", coarse_probabilities=[1, 0, 0, 0])
        self.assertEqual(neural._examples(modified, "configured entity", []), features)
        self.assertNotIn("SECRET_GOLD", repr(features))

    def test_source_and_selection_tampering_fail_checksums(self):
        original = self.weather.read_bytes()
        config = self.configured()
        self.weather.write_text(self.weather.read_text().replace("Raw weather text", "Changed weather"))
        with self.assertRaisesRegex(ValueError, "SHA-256|size"):
            load_auxiliary(config)
        self.weather.write_bytes(original)
        config = self.configured()
        config["data_dir"] = str(self.root / "fresh-selection-cache")
        self.selection.write_text(self.selection.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256|size"):
            load_auxiliary(config)

    def test_missing_duplicate_or_nontrain_selection_is_rejected(self):
        for change in ("missing", "duplicate", "test"):
            with self.subTest(change=change):
                original = copy.deepcopy(self.sources)
                if change == "missing":
                    self.sources["weather"]["items"][0]["id"] = "partly-sunny:999"
                elif change == "duplicate":
                    self.sources["weather"]["items"].append(copy.deepcopy(self.sources["weather"]["items"][0]))
                else:
                    self.sources["weather"]["split"] = "test"
                with self.assertRaises(ValueError):
                    load_auxiliary(self.configured())
                self.sources = original

    def test_replay_samples_one_combined_pool_and_preserves_flags(self):
        config = self.configured()
        source = self.root / "news.jsonl"
        source.write_text(json.dumps({"id": "human", "group_id": "news", "entity": "Target", "text": "Target text",
                                      "target_start": 0, "target_end": 6, "split": "train",
                                      "gold_relevant": True, "gold_polarity": "negative"}) + "\n")
        config["auxiliary_supervision"].update(path=str(source), sha256=self.asset(source)["sha256"])
        full, full_targets, full_metadata = load_auxiliary(config)
        expected = sorted(zip(full, full_targets), key=lambda pair: hashlib.sha256(f"42:2:{pair[0]['id'].removeprefix('aux:')}".encode()).digest())[:2]
        items, targets, metadata = load_auxiliary(config, mode="incremental", version=2, seed=42)
        self.assertEqual(items, [pair[0] for pair in expected])
        self.assertEqual(targets, [pair[1] for pair in expected])
        self.assertEqual(metadata["available_items"], 4)
        self.assertEqual(metadata["sha256"], full_metadata["sha256"])
        self.assertEqual((items, targets, metadata), load_auxiliary(config, mode="incremental", version=2, seed=42))

    def test_content_identity_is_cache_location_independent(self):
        config = self.configured()
        before = load_auxiliary(config)
        config["data_dir"] = str(self.root / "different-cache")
        with patch.dict("os.environ", {"MONITOR_DATA_DIR": str(self.root / "native-cache")}):
            after = load_auxiliary(config)
        self.assertEqual(before, after)
        self.assertTrue((self.root / "native-cache/training-sources").is_dir())
        self.assertFalse((self.root / "different-cache").exists())
