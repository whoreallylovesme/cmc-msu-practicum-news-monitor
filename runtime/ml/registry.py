"""Persist model revisions and their serialized artifacts in one SQLite transaction."""

import hashlib
import io
import json
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import joblib


class Registry:
    def __init__(self, path):
        self.path = path
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS service_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_artifacts (
                    model_id TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version > 0),
                    sha256 TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY (model_id, version)
                );
            """)

    def state(self):
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute("SELECT payload FROM service_state WHERE id = 1").fetchone()
        return json.loads(row[0]) if row else {"models": {}, "jobs": {}}

    def save(self, state, revision=None):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            if revision is not None:
                connection.execute(
                    "INSERT INTO model_artifacts VALUES (?, ?, ?, ?)",
                    revision,
                )
            connection.execute(
                """INSERT INTO service_state VALUES (1, ?)
                   ON CONFLICT (id) DO UPDATE SET payload = excluded.payload""",
                (json.dumps(state, ensure_ascii=False, allow_nan=False),),
            )

    @lru_cache(maxsize=2)
    def load(self, model_id, version):
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT payload, sha256 FROM model_artifacts WHERE model_id = ? AND version = ?",
                (model_id, version),
            ).fetchone()
        if row is None:
            raise ValueError("Model artifact not found")
        payload, checksum = row
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError("Model artifact checksum mismatch")
        return joblib.load(io.BytesIO(payload))

    def install_bundled(self, manifest_path):
        """Install a trusted, checksummed local model once; preserve user revisions."""
        path = Path(manifest_path)
        if not path.is_file():
            return
        document = json.loads(path.read_text(encoding="utf-8"))
        model = document["model"]
        state = self.state()
        if model["id"] in state["models"]:
            return
        payload = (path.parent / document["artifact"]).read_bytes()
        record = next(item for item in model["versions"] if item["version"] == model["version"])
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != record["artifact_sha256"] or len(payload) != record["artifact_bytes"]:
            raise ValueError("Bundled model checksum or size mismatch")
        state["models"][model["id"]] = model
        self.save(state, (model["id"], model["version"], checksum, payload))


def serialize(artifact):
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()
