"""Official SemEval TRAIN topics retain their real five-level human targets."""

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import training_sources
from lib.supervision import load_auxiliary


TRAIN_MEMBER = "2017_English_final/GOLD/Subtasks_CE/twitter-2016train-CE.txt"


class SemEvalSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "source.zip"
        self.selector = self.root / "selection.yaml"
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("MONITOR_DATA_DIR", None)
        self.lines = [
            f"{100 + index}\tGiven Topic\t{label}\tRaw café row{index}"
            + ("\tkept" if index == 3 else "")
            for index, label in enumerate(("-2", "-1", "0", "1", "2"))
        ]
        self.raw = ("\n".join(self.lines) + "\n").encode()
        self.write_zip(self.raw)
        topic_hash = hashlib.sha256(b"Given Topic").hexdigest()[:16]
        self.items = [
            {"id": f"semeval2016C:{100 + index}:{topic_hash}",
             "group_id": f"source-group:{index}"}
            for index in (4, 0, 3, 1, 2)
        ]
        self.selected = {
            "format": "semeval_topic_five_point_zip",
            "split": "train",
            "asset": self.asset(self.archive),
            "member": TRAIN_MEMBER,
            "member_sha256": hashlib.sha256(self.raw).hexdigest(),
            "member_size_bytes": len(self.raw),
            "items": self.items,
        }
        network = patch("requests.get", side_effect=AssertionError("Offline fixtures cannot use the network"))
        network.start()
        self.addCleanup(network.stop)

    @staticmethod
    def asset(path):
        return {
            "path": str(path), "is_local": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    def write_zip(self, raw):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(TRAIN_MEMBER, raw)
            archive.writestr(TRAIN_MEMBER.replace("train-CE", "dev-CE"), "NEVER READ DEV")
            archive.writestr(TRAIN_MEMBER.replace("train-CE", "test-CE"), "NEVER READ TEST")

    def configured(self):
        self.selected["asset"] = self.asset(self.archive)
        self.selector.write_text(yaml.safe_dump(
            {"version": 1, "sources": {"semeval": self.selected}}, sort_keys=False
        ))
        return {
            "data_dir": str(self.root / "cache"),
            "auxiliary_supervision": {
                "path": None,
                "additional_sources": [{
                    "name": "semeval", "format": "semeval_topic_five_point_zip",
                    "asset": self.asset(self.archive), "selection": self.asset(self.selector),
                }],
            },
        }

    def test_auxiliary_integration_preserves_real_five_levels_and_raw_inputs(self):
        items, targets, metadata = load_auxiliary(self.configured())
        self.assertEqual([item["id"] for item in items], ["aux:" + row["id"] for row in self.items])
        self.assertEqual([item["story_id"] for item in items], ["aux:" + row["group_id"] for row in self.items])
        self.assertEqual(targets, [
            [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
        ])
        for item in items:
            self.assertEqual(item["entity"], "Given Topic")
            self.assertIs(item["partial_labels"], False)
            self.assertEqual(item["sample_weight"], 1.0)
            self.assertEqual(item["parent_text"], "")
            self.assertEqual(item["story_title"], "")
            self.assertIsNone(item["parent_id"])
            self.assertEqual(set(item), {
                "id", "story_id", "entity", "text", "parent_text", "story_title",
                "parent_id", "partial_labels", "sample_weight",
            })
        self.assertEqual(items[0]["text"], "Raw café row4")
        self.assertEqual(items[2]["text"], "Raw café row3\tkept")
        self.assertEqual(metadata["available_items"], 5)
        self.assertEqual(metadata["items"], 5)
        self.assertEqual(metadata["selected_ids"], [item["id"] for item in items])
        source = metadata["sources"][0]
        self.assertEqual(source["format_version"], "selected-semeval-topic-five-v1")
        self.assertIs(source["known_related"], True)
        self.assertEqual(source["member"], TRAIN_MEMBER)
        self.assertEqual(source["member_sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(source["member_size_bytes"], len(self.raw))
        self.assertEqual(source["source_ids"], [row["id"] for row in self.items])
        self.assertEqual(source["group_ids"], [row["group_id"] for row in self.items])

    def test_only_pinned_train_member_is_opened_without_extraction(self):
        config = self.configured()
        opened = []
        original = zipfile.ZipFile.open

        def track(archive, member, *args, **kwargs):
            opened.append(member.filename if isinstance(member, zipfile.ZipInfo) else member)
            return original(archive, member, *args, **kwargs)

        with patch.object(zipfile.ZipFile, "open", track), patch.object(
            zipfile.ZipFile, "extractall", side_effect=AssertionError("Never extract archives")
        ):
            training_sources.load_sources(config)
        self.assertEqual(opened, [TRAIN_MEMBER])

    def test_non_training_member_is_rejected_before_open(self):
        members = [TRAIN_MEMBER.replace("train-CE", kind + "-CE") for kind in ("dev", "devtest", "test")]
        members.append(TRAIN_MEMBER.replace("/GOLD/", "/TEST/"))
        for member in members:
            with self.subTest(member=member):
                self.selected["member"] = member
                with patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("Never read non-TRAIN")):
                    with self.assertRaisesRegex(ValueError, "TRAIN"):
                        training_sources.load_sources(self.configured())

    def test_zip_and_member_checksums_and_size_are_verified(self):
        self.selected["member_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            training_sources.load_sources(self.configured())
        self.selected["member_sha256"] = hashlib.sha256(self.raw).hexdigest()
        self.selected["member_size_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "byte size"):
            training_sources.load_sources(self.configured())
        self.selected["member_size_bytes"] = len(self.raw)
        config = self.configured()
        payload = self.archive.read_bytes()
        self.archive.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        # Previously verified cached bytes remain valid; an empty cache must reject corruption.
        config["data_dir"] = str(self.root / "fresh-corrupt-source-cache")
        with self.assertRaisesRegex(ValueError, "SHA-256|size"):
            training_sources.load_sources(config)

    def test_unknown_label_missing_text_and_duplicate_source_id_fail(self):
        cases = {
            "unknown label": self.raw.replace(b"\t-2\t", b"\t3\t"),
            "missing text field": b"100\tGiven Topic\t-2\n",
            "empty text": self.raw.replace("Raw café row0".encode(), b""),
            "duplicate source ID": self.raw + self.lines[0].encode() + b"\n",
        }
        for name, raw in cases.items():
            with self.subTest(case=name):
                self.write_zip(raw)
                self.selected.update(member_sha256=hashlib.sha256(raw).hexdigest(), member_size_bytes=len(raw))
                with self.assertRaisesRegex(ValueError, "original human|raw text|repeats"):
                    training_sources.load_sources(self.configured())

    def test_missing_requested_tweet_topic_is_not_silently_dropped(self):
        self.items[0]["id"] = "semeval2016C:999999:missing"
        with self.assertRaisesRegex(ValueError, "missing selected"):
            training_sources.load_sources(self.configured())
