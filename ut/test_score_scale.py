"""The public score range and one-time migration use the same 0–1 contract."""

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.contracts import Prediction
from lib.sentiment import SCORE_LEVELS, SCORE_MAX, SCORE_MIN
from runtime.storage.database import Database, SCHEMA, UPSERT_COMMENT, UPSERT_LABEL, UPSERT_PREDICTION
from runtime.web_master import main as master
from runtime.visualization.dashboard import timeline_figure


SPEC = importlib.util.spec_from_file_location("demo_e2e_score_test", ROOT / "demo/e2e.py")
e2e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e2e)


class ScoreScaleTests(unittest.TestCase):
    def test_public_contract_and_demo_reject_scores_outside_unit_interval(self):
        self.assertEqual(SCORE_LEVELS, (0, .25, .5, .75, 1))
        scale = {"min": SCORE_MIN, "max": SCORE_MAX}
        self.assertEqual(scale, {"min": 0, "max": 1})
        prediction = {"comment_id": "c", "probabilities": [0, 0, 1, 0, 0, 0], "relevant": True}
        for score in (0, .5, 1):
            Prediction(**prediction, score=score)
            e2e.validate_prediction({**prediction, "score": score}, "c", scale)
        for score in (-.01, 1.01, 5):
            with self.subTest(score=score):
                with self.assertRaises(ValidationError):
                    Prediction(**prediction, score=score)
                with self.assertRaises(AssertionError):
                    e2e.validate_prediction({**prediction, "score": score}, "c", scale)

    def test_gateway_analytics_comments_and_plot_preserve_canonical_scores(self):
        model = {"id": "m", "version": 1, "entity": "ChatGPT"}
        counts = {"comments": 2, "mentions": 1, "scored": 2, "relevant": 1, "score": .25}
        period = {**counts, "date": "2026-01-01"}
        analytics = {"summary": counts, "periods": [period], "sources": [{**counts, "source": "hacker_news"}]}

        def upstream(service, _method, path, **_options):
            if service == "ml":
                return model
            if path == "/sources":
                return {"items": []}
            if path == "/predictions":
                return {"items": [{"comment_id": "c", "score": .25}], "total": 1}
            return analytics

        with patch.object(master, "upstream", side_effect=upstream), TestClient(master.app) as client:
            options = {"auth": master.CREDENTIALS["viewer"]}
            config = client.get("/api/config", **options).json()
            self.assertEqual(config["score_scale"], {"min": 0, "max": 1})
            result = client.get("/api/comments?entity=ChatGPT&model_id=m", **options)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["items"][0]["score"], .25)
            result = client.get("/api/analytics?entity=ChatGPT&model_id=m", **options)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json(), {**analytics, "score_scale": config["score_scale"]})
            figure = timeline_figure(result.json()["periods"], config["score_scale"])
            self.assertEqual(tuple(figure.layout.yaxis.range), (0, 1))
            self.assertEqual(figure.layout.shapes[0].y0, .5)

    def test_internal_ml_proxy_does_not_rescale_predictions_or_metrics(self):
        payload = {"items": [{"comment_id": "c", "score": .5}], "metrics": {"score_mae": .2}}
        with patch.object(master.httpx, "request", return_value=httpx.Response(200, json=payload)), \
                TestClient(master.app) as client:
            result = client.post("/services/ml/models/m/predict", json={"items": []},
                                 auth=master.CREDENTIALS["admin"])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), {**payload, "score_scale": {"min": 0, "max": 1}})


class StorageScaleMigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "archive.sqlite3"
        self.database = Database(self.path, 1)

    def seed(self, legacy=False):
        if legacy:
            with sqlite3.connect(self.path) as connection:
                connection.executescript(SCHEMA.replace("score BETWEEN 0 AND 1", "score BETWEEN 0 AND 5"))
        else:
            self.database.initialize()
        with self.database.connect(write=True) as connection:
            for comment_id, score in (("relevant", 2.5 if legacy else .5), ("irrelevant", None)):
                connection.execute(UPSERT_COMMENT, {
                    "entity": "ChatGPT", "id": comment_id, "story_id": "story", "parent_id": None,
                    "published_at": "2026-01-01T00:00:00Z", "story_title": "Original title",
                    "text": "Original comment", "parent_text": "Original parent", "url": "https://example.test",
                    "source": "hacker_news", "content_hash": "unchanged",
                })
                values = json.dumps([0, 0, 1, 0, 0, 0] if score is not None else [0, 0, 0, 0, 0, 1])
                connection.execute(UPSERT_PREDICTION, {"entity": "ChatGPT", "model_id": "m",
                    "comment_id": comment_id, "probabilities": values, "score": score, "relevant": score is not None})
                connection.execute(UPSERT_LABEL, {"entity": "ChatGPT", "teacher": "pinned",
                    "comment_id": comment_id, "probabilities": values, "input_hash": "original-input"})

    def snapshot(self):
        with self.database.connect() as connection:
            return {table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                    for table in ("comments", "labels", "predictions")}

    def test_legacy_migration_is_once_only_and_keeps_text_labels_probabilities(self):
        self.seed(legacy=True)
        before = self.snapshot()
        self.database.initialize()
        self.database.initialize()
        after = self.snapshot()
        self.assertEqual(after["comments"], before["comments"])
        self.assertEqual(after["labels"], before["labels"])
        self.assertEqual([row[:4] + row[5:] for row in after["predictions"]],
                         [row[:4] + row[5:] for row in before["predictions"]])
        self.assertEqual([row[4] for row in after["predictions"]], [.5, None])
        with self.database.connect() as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            schema = connection.execute("SELECT sql FROM sqlite_master WHERE name='predictions'").fetchone()[0]
            self.assertIn("score BETWEEN 0 AND 5", schema)

    def test_new_database_enforces_unit_range_and_does_not_rescale_on_restart(self):
        self.seed()
        with self.database.connect(write=True) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE predictions SET score=1.01 WHERE comment_id='relevant'")
            connection.execute("UPDATE predictions SET score=.75 WHERE comment_id='relevant'")
        self.database.initialize()
        self.assertEqual(self.snapshot()["predictions"][0][4], .75)

    def test_failed_migration_rolls_back_scores_and_version(self):
        self.seed(legacy=True)
        before = self.snapshot()
        with self.database.connect(write=True) as connection:
            connection.execute("CREATE TRIGGER fixture_abort BEFORE UPDATE OF score ON predictions "
                               "BEGIN SELECT RAISE(ABORT, 'fixture failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "fixture failure"):
            self.database.initialize()
        self.assertEqual(self.snapshot(), before)
        with self.database.connect() as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
