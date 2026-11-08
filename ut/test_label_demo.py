"""Full snapshot publication must survive interrupted labelling without a model."""

import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "demo/quality"), str(ROOT / "runtime/teacher")]

from label_demo import annotate, json_bytes, label_snapshot, source_context
import label_demo
from dataset_labels import fingerprint, load_dataset_snapshot
from lib.teacher_contracts import TeacherComment, input_hash


class FakeRuntime:
    def __init__(self, _settings=None, fail_after=None, delay=0):
        self.seen, self.fail_after, self.closed = [], fail_after, False
        self.lock, self.active, self.peak, self.delay = threading.Lock(), 0, 0, delay

    def distribution(self, entity, aliases, item):
        assert not any(key.startswith("label") or key == "fixture_label" for key in item.model_dump())
        with self.lock:
            if self.fail_after is not None and len(self.seen) >= self.fail_after:
                raise RuntimeError("Interrupted teacher")
            self.seen.append((entity, list(aliases), item.id))
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            values = [0.06] * 6
            values[int(item.id) % 6] = 0.7
            return values
        finally:
            with self.lock:
                self.active -= 1

    def close(self):
        self.closed = True


class LabelDemoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.full = self.directory / "10k.json"
        self.subset = self.directory / "1k.json"
        self.output = self.directory / "output"
        self.output.mkdir()
        self.settings = {"revision": "teacher-test", "protocol": {"prompt": "original"},
                         "model_name": "example/model", "model_revision": "pinned",
                         "model": SimpleNamespace(path="https://example.test/model.gguf", is_local=False,
                                                  sha256="a" * 64, size_bytes=100)}
        self.config = {"snapshot_path": str(self.full), "subset_path": str(self.subset),
                       "entity": "ChatGPT", "aliases": ["Chat GPT"]}

    @staticmethod
    def predict(runtime):
        return lambda entity, item, aliases: runtime.distribution(entity, aliases, TeacherComment(**item))

    def fixture(self):
        stories, subset = [], []
        for group in range(2):
            comments = []
            for offset in range(5000):
                index = group * 5000 + offset
                row = {"id": str(100000 + index), "story_id": str(group), "parent_id": str(group),
                       "story_title": "A real API-shaped thread", "text": "No entity name here\n☃ " + str(index),
                       "parent_text": "Контекст " * 200 if index % 17 == 0 else None,
                       "published_at": f"2024-01-01T00:{index % 60:02d}:00+00:00",
                       "url": f"https://news.ycombinator.com/item?id={100000 + index}", "source": "hacker_news",
                       "extra_raw_field": {"unicode": "текст", "items": [1, None, 0.25]},
                       "fixture_label": None, "label": "F", "label_entity": "ChatGPT", "label_source": "old_rule"}
                row["label_fingerprint"] = fingerprint(row)
                comments.append(row)
            story = {"title": "Thread", "date": "2024-01-01T00:00:00+00:00", "comments": comments}
            stories.append(story)
            subset.append({**story, "comments": copy.deepcopy(comments[::10])})
        previous = {"source_api": "https://hn.algolia.com/api/v1", "attribution": "Original HN authors",
                    "collection": {"selection": "unchanged historical collection"},
                    "matching": {"allow_typos": True}, "mention_candidate_count": 12,
                    "prepared_labels": {"sources": {"old_rule": 10000}, "new_candidate_teacher": "old_model"}}
        for path, value in ((self.full, stories), (self.subset, subset),
                            (self.full.with_suffix(".metadata.json"), previous),
                            (self.subset.with_suffix(".metadata.json"), {**previous, "subset_of": {"method": "old quotas"}})):
            path.write_bytes(json_bytes(value))
        return stories, subset

    def test_interrupted_run_preserves_originals_and_resume_publishes_exact_subset(self):
        original, original_subset = self.fixture()
        paths = (self.full, self.subset, self.full.with_suffix(".metadata.json"), self.subset.with_suffix(".metadata.json"))
        before = {path: path.read_bytes() for path in paths}
        interrupted = FakeRuntime(fail_after=13)
        with self.assertRaisesRegex(RuntimeError, "Interrupted teacher"):
            label_snapshot(self.config, self.settings, self.predict(interrupted), self.output)
        self.assertEqual({path: path.read_bytes() for path in paths}, before)
        resumed = FakeRuntime()
        result = label_snapshot(self.config, self.settings, self.predict(resumed), self.output)
        self.assertEqual(result["cached_items"], 13)
        self.assertEqual(len(resumed.seen), 9987)
        full, subset = json.loads(self.full.read_bytes()), json.loads(self.subset.read_bytes())
        self.assertEqual(source_context(full), source_context(original))
        self.assertEqual(source_context(subset), source_context(original_subset))
        indexed = {row["id"]: row for story in full for row in story["comments"]}
        selected = [row for story in subset for row in story["comments"]]
        self.assertEqual([row["id"] for row in selected], [row["id"] for story in original_subset for row in story["comments"]])
        self.assertTrue(all(row == indexed[row["id"]] for row in selected))
        for row in indexed.values():
            self.assertNotIn("fixture_label", row)
            self.assertEqual(row["label_fingerprint"], fingerprint(row))
            self.assertEqual(row["label_input_hash"], input_hash("ChatGPT", row, ["Chat GPT"]))
            self.assertEqual(row["label_source"], "teacher-test")
        loaded = load_dataset_snapshot(self.full, "ChatGPT")
        self.assertEqual(len(loaded["labels"]), 10000)
        self.assertEqual(len(load_dataset_snapshot(self.subset, "ChatGPT")["labels"]), 1000)
        full_meta, subset_meta = (json.loads(path.read_bytes()) for path in paths[2:])
        self.assertEqual(full_meta["sha256"], result["sha256"])
        self.assertEqual(full_meta["subsets"][0]["sha256"], result["subset_sha256"])
        self.assertEqual(subset_meta["subset_of"]["sha256"], result["sha256"])
        self.assertEqual(full_meta["collection"], {"selection": "unchanged historical collection"})
        self.assertNotIn("matching", full_meta)
        self.assertNotIn("mention_candidate_count", full_meta)
        self.assertEqual(full_meta["prepared_labels"]["sources"], {"teacher-test": 10000})
        self.assertEqual(full_meta["prepared_labels"]["teacher"]["model_asset"]["sha256"], "a" * 64)
        published = {path: path.read_bytes() for path in paths}
        cached_runtime = FakeRuntime(fail_after=0)
        rerun = label_snapshot(self.config, self.settings, self.predict(cached_runtime), self.output)
        self.assertEqual(rerun["cached_items"], 10000)
        self.assertEqual({path: path.read_bytes() for path in paths}, published)

    def test_cache_identity_covers_aliases_full_context_protocol_and_revision(self):
        row = {"id": "1", "story_id": "2", "text": "No entity", "story_title": "Title",
               "parent_text": "x" * 12000, "parent_id": "2"}
        stories = [{"comments": [row]}]
        runtime = FakeRuntime()
        cache = self.output / "labels.sqlite3"
        for aliases, settings, text in ((["Chat GPT"], self.settings, stories),
                                       (["Chat GPT"], self.settings, stories),
                                       (["GPT chat"], self.settings, stories),
                                       (["Chat GPT"], {**self.settings, "revision": "new"}, stories),
                                       (["Chat GPT"], {**self.settings, "protocol": {"prompt": "new"}}, stories),
                                       (["Chat GPT"], self.settings, [{"comments": [{**row, "parent_text": row["parent_text"] + "tail"}]}])):
            annotate(text, "ChatGPT", aliases, self.predict(runtime), settings, cache)
        self.assertEqual(len(runtime.seen), 5)

    def test_parallel_snapshot_is_bounded_resumable_and_matches_serial_bytes(self):
        original, original_subset = self.fixture()
        paths = (self.full, self.subset, self.full.with_suffix(".metadata.json"), self.subset.with_suffix(".metadata.json"))
        before = {path: path.read_bytes() for path in paths}
        settings = {**self.settings, "llama": {"parallel": 4}}
        instances = []

        class CountingExecutor(ThreadPoolExecutor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.submitted = 0
                instances.append(self)

            def submit(self, *args, **kwargs):
                self.submitted += 1
                return super().submit(*args, **kwargs)

        interrupted = FakeRuntime(fail_after=13, delay=.0001)
        with patch.object(label_demo, "ThreadPoolExecutor", CountingExecutor):
            with self.assertRaisesRegex(RuntimeError, "Interrupted teacher"):
                label_snapshot(self.config, settings, self.predict(interrupted), self.output)
        self.assertLessEqual(instances[0].submitted, 17)
        self.assertEqual(interrupted.active, 0)
        self.assertEqual({path: path.read_bytes() for path in paths}, before)
        resumed = FakeRuntime(delay=.0001)
        result = label_snapshot(self.config, settings, self.predict(resumed), self.output)
        self.assertGreater(result["cached_items"], 0)
        self.assertLessEqual(result["cached_items"], 13)
        self.assertEqual(len(resumed.seen), 10000 - result["cached_items"])
        self.assertEqual(len({row[2] for row in resumed.seen}), len(resumed.seen))
        self.assertGreater(resumed.peak, 1)
        self.assertLessEqual(resumed.peak, 4)
        full, subset = json.loads(self.full.read_bytes()), json.loads(self.subset.read_bytes())
        indexed = {row["id"]: row for story in full for row in story["comments"]}
        self.assertEqual(source_context(full), source_context(original))
        self.assertEqual(source_context(subset), source_context(original_subset))
        self.assertTrue(all(row == indexed[row["id"]] for story in subset for row in story["comments"]))
        self.assertEqual(len(load_dataset_snapshot(self.full, "ChatGPT")["labels"]), 10000)
        self.assertEqual(len(load_dataset_snapshot(self.subset, "ChatGPT")["labels"]), 1000)
        published = {path: path.read_bytes() for path in paths}
        # A fresh parallel run calls the same fake model exactly once per input.
        fresh_output = self.directory / "fresh-output"
        fresh = FakeRuntime(delay=.0001)
        fresh_result = label_snapshot(self.config, settings, self.predict(fresh), fresh_output)
        self.assertEqual(fresh_result["cached_items"], 0)
        self.assertEqual(len(fresh.seen), 10000)
        self.assertEqual(len({row[2] for row in fresh.seen}), 10000)
        self.assertEqual({path: path.read_bytes() for path in paths}, published)
        # Serial and parallel execution share the unchanged cache/provenance key.
        cached = FakeRuntime(fail_after=0)
        rerun = label_snapshot(self.config, self.settings, self.predict(cached), fresh_output)
        self.assertEqual(rerun["cached_items"], 10000)
        self.assertEqual({path: path.read_bytes() for path in paths}, published)

    def test_parallel_callback_initializes_and_closes_one_runtime(self):
        spec = importlib.util.spec_from_file_location("quality_parallel_main_under_test", ROOT / "demo/quality/main.py")
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)
        import model
        runtime = FakeRuntime(delay=.01)

        def construct(_settings):
            time.sleep(.02)
            return runtime

        with patch.object(model, "load_settings", return_value=self.settings), patch.object(model, "TeacherRuntime", side_effect=construct) as constructor:
            with main.teacher({"teacher_config": "ignored.yaml"}) as (_, predict):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(predict, "ChatGPT", {"id": str(index), "story_id": "2", "text": "No direct mention"})
                               for index in range(4)]
                    for future in futures:
                        self.assertAlmostEqual(sum(future.result()), 1)
                self.assertEqual(constructor.call_count, 1)
                self.assertGreater(runtime.peak, 1)
                self.assertFalse(runtime.closed)
        self.assertTrue(runtime.closed)

    def test_main_closes_owned_runtime_on_error(self):
        spec = importlib.util.spec_from_file_location("quality_main_under_test", ROOT / "demo/quality/main.py")
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)
        import model
        runtime = FakeRuntime()
        with patch.object(model, "load_settings", return_value=self.settings), patch.object(model, "TeacherRuntime", return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, "downstream"):
                with main.teacher({"teacher_config": "ignored.yaml"}) as (_, predict):
                    predict("ChatGPT", {"id": "1", "story_id": "2", "text": "No direct mention"})
                    raise RuntimeError("downstream")
        self.assertTrue(runtime.closed)
