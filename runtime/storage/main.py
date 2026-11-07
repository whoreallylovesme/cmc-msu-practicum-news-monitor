import hashlib
import json
import logging
import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import monotonic
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from lib.config import load_config
from lib.contracts import Comment as CommentRecord
from lib.contracts import Identifier, Prediction, Probabilities
from lib.mentions import MentionMatcher

from database import (
    PREDICTIONS_JOIN,
    PREDICTIONS_SELECT,
    UPSERT_COMMENT,
    UPSERT_LABEL,
    UPSERT_PREDICTION,
    Database,
)


class Settings(BaseModel):
    host: str
    port: int
    data_dir: Path
    database: Path
    log: Path
    busy_timeout_seconds: float = Field(gt=0)
    log_max_bytes: int = Field(gt=0)
    log_backups: int = Field(ge=1)


class Comment(CommentRecord):
    def storage_row(self, entity: str) -> dict:
        record = self.model_dump()
        record["published_at"] = utc_iso(self.published_at)
        record["entity"] = entity
        context = [self.story_id, self.story_title, self.parent_id, self.parent_text, self.text]
        record["content_hash"] = hashlib.sha256(json.dumps(context, ensure_ascii=False).encode("utf-8")).hexdigest()
        return record


class CommentBatch(BaseModel):
    entity: Identifier
    items: list[Comment] = Field(max_length=5000)


class PredictionBatch(BaseModel):
    entity: Identifier
    model_id: Identifier
    items: list[Prediction] = Field(max_length=5000)


class Label(Probabilities):
    input_hash: Optional[str] = Field(default=None, max_length=200)


class LabelBatch(BaseModel):
    entity: Identifier
    teacher: Identifier
    items: list[Label] = Field(max_length=5000)


def utc_iso(timestamp: datetime) -> str:
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


def load_settings() -> Settings:
    config = load_config(
        Path(__file__).with_name("config.yaml"),
        env_var="STORAGE_CONFIG",
        overrides={"data_dir": "MONITOR_DATA_DIR"},
    )
    return Settings(**config)


def parse_prediction(row: sqlite3.Row) -> dict:
    record = dict(row)
    record["probabilities"] = json.loads(record["probabilities"])
    record["relevant"] = bool(record["relevant"])
    return record


def empty_counts() -> dict:
    return {"comments": 0, "mentions": 0, "direct_mentions": 0, "typo_mentions": 0,
            "context_mentions": 0, "scored": 0, "relevant": 0, "score": 0.0}


def average_score(counts: dict) -> dict:
    return {
        **counts,
        "score": counts["score"] / counts["relevant"] if counts["relevant"] else None,
    }


def comment_filters(query, from_date, to_date):
    if from_date is not None and to_date is not None and from_date >= to_date:
        raise HTTPException(422, "from_date must be earlier than to_date")
    clauses, parameters = [], []
    if query:
        words = re.findall(r"[^\W_]+", query)
        if words:
            clauses.append("c.rowid IN (SELECT rowid FROM comment_search WHERE comment_search MATCH ?)")
            parameters.append(" AND ".join(f'"{word}"' for word in words))
        else:
            clauses.append("0")
    if from_date is not None:
        clauses.append("c.published_at >= ?")
        parameters.append(from_date.isoformat())
    if to_date is not None:
        clauses.append("c.published_at < ?")
        parameters.append(to_date.isoformat())
    return "".join(" AND " + clause for clause in clauses), parameters


def check_content(
    connection: sqlite3.Connection, entity: str, items: list[Probabilities]
) -> None:
    for item in items:
        if item.content_hash is None:
            continue
        row = connection.execute(
            "SELECT content_hash FROM comments WHERE entity = ? AND id = ?",
            (entity, item.comment_id),
        ).fetchone()
        if row is None or row["content_hash"] != item.content_hash:
            raise HTTPException(409, f"Comment {item.comment_id} was changed or deleted")


def create_app(settings: Settings) -> FastAPI:
    database = Database(settings.data_dir / settings.database, settings.busy_timeout_seconds)
    access_logger = logging.Logger("storage.access", level=logging.INFO)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        database.initialize()
        log_path = settings.data_dir / settings.log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_path, maxBytes=settings.log_max_bytes, backupCount=settings.log_backups, encoding="utf-8"
        )
        access_logger.addHandler(handler)
        try:
            yield
        finally:
            access_logger.removeHandler(handler)
            handler.close()

    service = FastAPI(title="News Monitor Storage", lifespan=lifespan)

    @service.middleware("http")
    async def log_requests(request: Request, call_next):
        start = monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            access_logger.info(json.dumps({
                "time": utc_iso(datetime.now(timezone.utc)),
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": round((monotonic() - start) * 1000, 2),
            }))

    @service.exception_handler(sqlite3.Error)
    async def database_error(_request: Request, error: sqlite3.Error):
        if isinstance(error, sqlite3.IntegrityError):
            return JSONResponse({"detail": "Record violates a storage constraint"}, status_code=409)
        if isinstance(error, sqlite3.OperationalError) and re.search(r"\blocked\b|\bbusy\b", str(error)):
            return JSONResponse({"detail": "Storage is temporarily busy"}, status_code=503)
        logging.getLogger("uvicorn.error").error("Unexpected SQLite error", exc_info=error)
        return JSONResponse({"detail": "Storage operation failed"}, status_code=500)

    @service.get("/health", include_in_schema=False)
    def health():
        with database.connect() as connection:
            connection.execute("SELECT 1 FROM comments LIMIT 1").fetchone()
        return {"status": "ok", "service": "storage"}

    @service.post("/comments/batch")
    def save_comments(batch: CommentBatch):
        rows = [item.storage_row(batch.entity) for item in batch.items]
        with database.connect(write=True) as connection:
            connection.executemany(UPSERT_COMMENT, rows)
        return {"stored": len(rows)}

    @service.get("/comments")
    def list_comments(
        entity: Identifier,
        limit: int = Query(default=100, ge=1, le=5000),
        offset: int = Query(default=0, ge=0),
        query: Optional[str] = Query(default=None, max_length=500),
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ):
        from_sql = "FROM comments c WHERE c.entity = ?"
        parameters = [entity]
        filters, filter_parameters = comment_filters(query, from_date, to_date)
        from_sql += filters
        parameters.extend(filter_parameters)
        with database.connect() as connection:
            connection.execute("BEGIN")
            total = connection.execute("SELECT COUNT(*) " + from_sql, parameters).fetchone()[0]
            rows = connection.execute(
                "SELECT c.* " + from_sql + " ORDER BY c.published_at DESC, c.id LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return {"items": [dict(row) for row in rows], "total": total}

    @service.get("/comments/{comment_id}")
    def get_comment(comment_id: Identifier, entity: Identifier):
        with database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM comments WHERE entity = ? AND id = ?", (entity, comment_id)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Comment not found")
        return dict(row)

    @service.put("/comments/{comment_id}")
    def put_comment(comment_id: Identifier, entity: Identifier, comment: Comment):
        if comment.id != comment_id:
            raise HTTPException(422, "Path and payload identify different comments")
        record = comment.storage_row(entity)
        with database.connect(write=True) as connection:
            connection.execute(UPSERT_COMMENT, record)
        return record

    @service.delete("/comments/{comment_id}")
    def delete_comment(comment_id: Identifier, entity: Identifier):
        with database.connect(write=True) as connection:
            cursor = connection.execute("DELETE FROM comments WHERE entity = ? AND id = ?", (entity, comment_id))
            if not cursor.rowcount:
                raise HTTPException(404, "Comment not found")
        return {"deleted": True}

    @service.post("/predictions/batch")
    def save_predictions(batch: PredictionBatch):
        rows = [
            {
                "entity": batch.entity,
                "model_id": batch.model_id,
                "comment_id": item.comment_id,
                "probabilities": json.dumps(item.probabilities),
                "score": item.score,
                "relevant": item.relevant,
            }
            for item in batch.items
        ]
        with database.connect(write=True) as connection:
            check_content(connection, batch.entity, batch.items)
            connection.executemany(UPSERT_PREDICTION, rows)
        return {"stored": len(rows)}

    @service.get("/predictions")
    def list_predictions(
        entity: Identifier,
        model_id: Identifier,
        limit: int = Query(default=100, ge=1, le=5000),
        offset: int = Query(default=0, ge=0),
        query: Optional[str] = Query(default=None, max_length=500),
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ):
        from_sql = PREDICTIONS_JOIN + " WHERE p.entity = ? AND p.model_id = ?"
        parameters = [entity, model_id]
        filters, filter_parameters = comment_filters(query, from_date, to_date)
        from_sql += filters
        parameters.extend(filter_parameters)
        with database.connect() as connection:
            connection.execute("BEGIN")
            total = connection.execute("SELECT COUNT(*) " + from_sql, parameters).fetchone()[0]
            rows = connection.execute(
                PREDICTIONS_SELECT + from_sql + " ORDER BY c.published_at DESC, c.id LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return {"items": [parse_prediction(row) for row in rows], "total": total}

    @service.get("/analytics")
    def analytics(
        entity: Identifier,
        model_id: Optional[Identifier] = None,
        bucket: Literal["day", "week"] = "week",
        aliases: list[str] = Query(default=[], max_length=10),
        allow_typos: bool = True,
        allow_context: bool = True,
        context_hints: list[str] = Query(default=[], max_length=20),
    ):
        """Count clear mentions with the same lightweight policy used by the model."""
        matcher = MentionMatcher(entity, aliases, {
            "allow_typos": allow_typos, "allow_context": allow_context, "context_hints": context_hints,
        })
        period = "date(c.published_at)" if bucket == "day" else "date(c.published_at, '-6 days', 'weekday 1')"
        summary = empty_counts()
        periods = {}
        sources = {}
        with database.connect() as connection:
            rows = connection.execute(
                f"SELECT c.id, c.text, c.story_id, c.parent_id, c.parent_text, c.story_title, c.source, {period} AS date,"
                " p.comment_id, p.relevant, p.score"
                " FROM comments c LEFT JOIN predictions p"
                " ON p.entity = c.entity AND p.comment_id = c.id AND p.model_id = ?"
                " WHERE c.entity = ?",
                (model_id, entity),
            )
            for row in rows:
                kind = matcher.match(dict(row))
                mention = kind is not None
                scored = row["comment_id"] is not None
                relevant = bool(row["relevant"])
                counts = (
                    summary,
                    periods.setdefault(row["date"], empty_counts()),
                    sources.setdefault(row["source"], empty_counts()),
                )
                for counter in counts:
                    counter["comments"] += 1
                    counter["mentions"] += mention
                    counter["direct_mentions"] += kind in {"direct", "alias"}
                    counter["typo_mentions"] += kind == "typo"
                    counter["context_mentions"] += kind == "context"
                    counter["scored"] += scored
                    counter["relevant"] += relevant
                    counter["score"] += row["score"] or 0
        return {
            "summary": average_score(summary),
            "periods": [
                {"date": day, **average_score(counts)}
                for day, counts in sorted(periods.items())
            ],
            "sources": [
                {"source": source, **average_score(counts)}
                for source, counts in sorted(
                    sources.items(), key=lambda item: (-item[1]["mentions"], item[0])
                )
            ],
        }

    @service.post("/labels/batch")
    def save_labels(batch: LabelBatch):
        rows = [
            {
                "entity": batch.entity,
                "teacher": batch.teacher,
                "comment_id": item.comment_id,
                "probabilities": json.dumps(item.probabilities),
                "input_hash": item.input_hash,
            }
            for item in batch.items
        ]
        with database.connect(write=True) as connection:
            check_content(connection, batch.entity, batch.items)
            connection.executemany(UPSERT_LABEL, rows)
        return {"stored": len(rows)}

    @service.get("/labels")
    def list_labels(
        entity: Identifier,
        teacher: Identifier,
        limit: int = Query(default=100, ge=1, le=5000),
        offset: int = Query(default=0, ge=0),
    ):
        where_sql = " WHERE l.entity = ? AND l.teacher = ?"
        with database.connect() as connection:
            connection.execute("BEGIN")
            total = connection.execute(
                "SELECT COUNT(*) FROM labels l" + where_sql, (entity, teacher)
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT l.*, c.content_hash FROM labels l"
                " JOIN comments c ON c.entity = l.entity AND c.id = l.comment_id"
                + where_sql + " ORDER BY l.comment_id LIMIT ? OFFSET ?",
                (entity, teacher, limit, offset),
            ).fetchall()
        items = [dict(row) for row in rows]
        for item in items:
            item["probabilities"] = json.loads(item["probabilities"])
        return {"items": items, "total": total}

    return service


settings = load_settings()
app = create_app(settings)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
