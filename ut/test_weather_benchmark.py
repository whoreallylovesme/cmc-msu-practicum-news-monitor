import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo/quality"))
import data


class WeatherBenchmarkTests(unittest.TestCase):
    def test_reconstructs_raw_context_known_confidence_and_off_topic_gold(self):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["id", "tweet", "s1", "s2", "s3", "s4", "s5"])
        writer.writeheader()
        writer.writerows([
            {"id": "1", "tweet": "Raw &gt; rain\nsecond line", "s1": .4, "s2": .6, "s3": 0, "s4": 0, "s5": 0},
            {"id": "2", "tweet": "A different subject", "s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 1},
        ])
        raw = buffer.getvalue().encode()
        expected = []
        for split, number, text, probabilities, weight, relevant, polarity in (
            ("dev", "1", "Raw &gt; rain\nsecond line", [1., 0., 0., 0.], .6, True, "negative"),
            ("test", "2", "A different subject", [0., 0., 0., 1.], 1., False, None),
        ):
            expected.append({"id": "partly-sunny:" + number, "group_id": "group-" + number,
                             "entity": "weather conditions", "text": text, "coarse_probabilities": probabilities,
                             "sample_weight": weight, "split": split, "dataset": "weather_adaptation",
                             "story_title": "", "parent_text": "", "gold_relevant": relevant,
                             "gold_polarity": polarity, "gold_intensity": None, "source_known_mass": weight})
        descriptor = {"version": 1, "dataset": "weather_adaptation", "entity": "weather conditions",
                      "source": {"url": "https://example.invalid/pinned.csv", "sha256": hashlib.sha256(raw).hexdigest(),
                                 "size_bytes": len(raw), "license": "LOCAL ONLY"},
                      "reference": "human votes", "split_policy": "fixed groups", "splits": {}}
        for row in expected:
            payload = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
            descriptor["splits"][row["split"]] = {"canonical_sha256": hashlib.sha256(payload).hexdigest(),
                                                   "items": [{"id": row["id"], "group_id": row["group_id"]}]}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            selector = directory / "selectors.yaml"
            selector.write_text(yaml.safe_dump(descriptor))
            with patch.object(data, "fetch", return_value=raw):
                result = data.prepare_weather_adaptation(directory, selector)
            (directory / "manifest.json").write_text(json.dumps({"datasets": [result]}))
            self.assertEqual(data.read_split(directory, "weather_adaptation", "dev"), [expected[0]])
            self.assertEqual(data.read_split(directory, "weather_adaptation", "test"), [expected[1]])

    def test_unknown_human_majority_cannot_become_neutral_or_off_topic(self):
        row = {"id": "1", "tweet": "Uncertain", "s1": ".6", "s2": ".1", "s3": ".1", "s4": ".1", "s5": ".1"}
        with self.assertRaises(AssertionError):
            data.canonical_adaptation(row, {"id": "partly-sunny:1", "group_id": "one"}, "dev", "weather conditions")
