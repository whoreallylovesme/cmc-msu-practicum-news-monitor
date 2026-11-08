"""Validate reproducible human diagnostic splits using the optional source cache."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("airline_preparation", ROOT / "demo/quality/data.py")
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)
CACHE = ROOT / ".local-data/quality/benchmarks/raw"


class AirlinePreparationTests(unittest.TestCase):
    @unittest.skipUnless(all((CACHE / name).is_file() for name in ("airline-Tweets.csv", "airline-README.md")),
                         "Optional source cache: run demo/quality operation prepare first")
    def test_cached_source_is_reproducible_and_groups_never_cross(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "raw").mkdir()
            for name in ("airline-Tweets.csv", "airline-README.md"):
                shutil.copyfile(CACHE / name, directory / "raw" / name)
            with patch("urllib.request.urlopen", side_effect=AssertionError("Cached preparation must not use the network")):
                metadata = data.prepare_airline(directory)
            self.assertEqual(metadata["curation"]["excluded_source_target_buckets"], {"Delta": 2222})
            self.assertEqual(metadata["curation"]["exact_duplicate_annotations_removed"], 137)
            expected = {
                "dev": (300, "49f39f18a69dcbbcfa382322eeb9dcaa46d2aabad3c81944b148a4221fe246e0"),
                "test": (1000, "30869b68bd13feed3f8c1f43915a2c5e0d5cea07ff62e295a31d25fa315a8235"),
            }
            rows = {}
            for split, (count, checksum) in expected.items():
                descriptor = metadata["files"]["airline_sentiment_" + split]
                self.assertEqual((descriptor["rows"], descriptor["sha256"]), (count, checksum))
                rows[split] = [json.loads(line) for line in (directory / descriptor["path"]).read_text().splitlines()]
                self.assertTrue(all(row["split"] == split and row["gold_relevant"] is True for row in rows[split]))
                self.assertTrue(all("target_start" not in row and "_author" not in row for row in rows[split]))
            self.assertEqual(metadata["files"]["airline_sentiment_dev"]["polarity_counts"], {"positive": 100, "neutral": 100, "negative": 100})
            self.assertEqual(metadata["files"]["airline_sentiment_dev"]["confidence"]["below_0_5"], 6)
            for key in ("group_id", "text_sha256", "source_id"):
                self.assertFalse({row[key] for row in rows["dev"]} & {row[key] for row in rows["test"]})
