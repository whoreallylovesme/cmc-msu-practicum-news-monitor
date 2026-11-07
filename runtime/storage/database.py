import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS comments (
    entity TEXT NOT NULL,
    id TEXT NOT NULL,
    story_id TEXT NOT NULL,
    parent_id TEXT,
    published_at TEXT NOT NULL,
    story_title TEXT NOT NULL,
    text TEXT NOT NULL,
    parent_text TEXT,
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    fixture_label INTEGER CHECK (fixture_label BETWEEN 0 AND 5),
    content_hash TEXT NOT NULL,
    PRIMARY KEY (entity, id)
);
CREATE INDEX IF NOT EXISTS comments_entity_date
    ON comments (entity, published_at DESC, id);
CREATE VIRTUAL TABLE IF NOT EXISTS comment_search USING fts5(
    story_title, text, parent_text, content='comments', content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS predictions (
    entity TEXT NOT NULL,
    model_id TEXT NOT NULL,
    comment_id TEXT NOT NULL,
    probabilities TEXT NOT NULL,
    score REAL CHECK (score BETWEEN 0 AND 5),
    relevant INTEGER NOT NULL CHECK (relevant IN (0, 1)),
    PRIMARY KEY (entity, model_id, comment_id),
    CHECK ((relevant = 1 AND score IS NOT NULL) OR (relevant = 0 AND score IS NULL)),
    FOREIGN KEY (entity, comment_id) REFERENCES comments (entity, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS predictions_comment ON predictions (entity, comment_id);

CREATE TABLE IF NOT EXISTS labels (
    entity TEXT NOT NULL,
    teacher TEXT NOT NULL,
    comment_id TEXT NOT NULL,
    probabilities TEXT NOT NULL,
    input_hash TEXT,
    PRIMARY KEY (entity, teacher, comment_id),
    FOREIGN KEY (entity, comment_id) REFERENCES comments (entity, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS labels_comment ON labels (entity, comment_id);

CREATE TRIGGER IF NOT EXISTS comments_search_insert AFTER INSERT ON comments BEGIN
    INSERT INTO comment_search(rowid, story_title, text, parent_text)
    VALUES (NEW.rowid, NEW.story_title, NEW.text, NEW.parent_text);
END;
CREATE TRIGGER IF NOT EXISTS comments_search_delete AFTER DELETE ON comments BEGIN
    INSERT INTO comment_search(comment_search, rowid, story_title, text, parent_text)
    VALUES ('delete', OLD.rowid, OLD.story_title, OLD.text, OLD.parent_text);
END;
CREATE TRIGGER IF NOT EXISTS comments_search_update
AFTER UPDATE OF story_title, text, parent_text ON comments
WHEN OLD.story_title IS NOT NEW.story_title OR OLD.text IS NOT NEW.text
     OR OLD.parent_text IS NOT NEW.parent_text BEGIN
    INSERT INTO comment_search(comment_search, rowid, story_title, text, parent_text)
    VALUES ('delete', OLD.rowid, OLD.story_title, OLD.text, OLD.parent_text);
    INSERT INTO comment_search(rowid, story_title, text, parent_text)
    VALUES (NEW.rowid, NEW.story_title, NEW.text, NEW.parent_text);
END;
CREATE TRIGGER IF NOT EXISTS comments_invalidate_scores
AFTER UPDATE OF content_hash ON comments
WHEN OLD.content_hash <> NEW.content_hash BEGIN
    DELETE FROM labels WHERE entity = OLD.entity AND comment_id = OLD.id;
    DELETE FROM predictions WHERE entity = OLD.entity AND comment_id = OLD.id;
END;
"""

UPSERT_COMMENT = """
INSERT INTO comments (entity, id, story_id, parent_id, published_at, story_title,
                      text, parent_text, url, source, fixture_label, content_hash)
VALUES (:entity, :id, :story_id, :parent_id, :published_at, :story_title,
        :text, :parent_text, :url, :source, :fixture_label, :content_hash)
ON CONFLICT (entity, id) DO UPDATE SET
    story_id = excluded.story_id,
    parent_id = excluded.parent_id,
    published_at = excluded.published_at,
    story_title = excluded.story_title,
    text = excluded.text,
    parent_text = excluded.parent_text,
    url = excluded.url,
    source = excluded.source,
    fixture_label = excluded.fixture_label,
    content_hash = excluded.content_hash
"""

UPSERT_PREDICTION = """
INSERT INTO predictions (entity, model_id, comment_id, probabilities, score, relevant)
VALUES (:entity, :model_id, :comment_id, :probabilities, :score, :relevant)
ON CONFLICT (entity, model_id, comment_id) DO UPDATE SET
    probabilities = excluded.probabilities,
    score = excluded.score,
    relevant = excluded.relevant
"""

UPSERT_LABEL = """
INSERT INTO labels (entity, teacher, comment_id, probabilities, input_hash)
VALUES (:entity, :teacher, :comment_id, :probabilities, :input_hash)
ON CONFLICT (entity, teacher, comment_id) DO UPDATE SET
    probabilities = excluded.probabilities,
    input_hash = excluded.input_hash
"""

PREDICTIONS_JOIN = """
FROM predictions p JOIN comments c ON c.entity = p.entity AND c.id = p.comment_id
"""

PREDICTIONS_SELECT = """
SELECT c.*, p.model_id, p.comment_id, p.probabilities, p.score, p.relevant
"""


class Database:
    def __init__(self, path: Path, timeout: float) -> None:
        self.path = path
        self.timeout = timeout

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript("BEGIN IMMEDIATE;\n" + SCHEMA + "\nCOMMIT;")

    @contextmanager
    def connect(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
