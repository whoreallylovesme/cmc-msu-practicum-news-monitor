"""HTTP API for importing batches of Hacker News comments."""

import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from lib.config import load_config, load_project_config
from lib.contracts import Comment, ProviderName
from lib.news import NewsProvider, comments_as_dto

from sources import ArchiveSource, HackerNewsSource, aliases_for


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("COLLECTOR_CONFIG", BASE_DIR / "config.yaml")).resolve()
CONFIG = load_config(CONFIG_PATH)
PROJECT = load_project_config()
DATA_DIR = Path(os.environ.get("MONITOR_DATA_DIR", CONFIG_PATH.parent / CONFIG["data_dir"]))
LOGGER = logging.getLogger(__name__)


class BatchRequest(BaseModel):
    entity: str = Field(min_length=1, max_length=100)
    aliases: List[str] = Field(default_factory=list, max_length=10)
    source: ProviderName = "hn_snapshot"
    cursor: Optional[str] = Field(default=None, max_length=512)
    limit: int = Field(default=50, ge=1, le=200)

    @field_validator("entity")
    @classmethod
    def check_entity(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("entity cannot be blank")
        return value

    @field_validator("aliases")
    @classmethod
    def check_aliases(cls, values):
        values = [value.strip() for value in values]
        if any(not value or len(value) > 100 for value in values):
            raise ValueError("aliases must contain 1 to 100 characters")
        return values


class BatchResponse(BaseModel):
    items: List[Comment]
    next_cursor: Optional[str]
    source: str


class LoadJournal:
    def __init__(self, path):
        self.path = path
        self.lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry):
        try:
            with self.lock, self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            LOGGER.exception("Could not write the collector journal")

    def recent(self, limit):
        entries = deque(maxlen=limit)
        with self.lock:
            try:
                with self.path.open(encoding="utf-8") as journal:
                    for line in journal:
                        try:
                            entries.append(json.loads(line))
                        except ValueError:
                            LOGGER.warning("Skipped a damaged journal record")
            except FileNotFoundError:
                return []
        return list(reversed(entries))


SOURCES: dict[str, NewsProvider] = {}


def register_provider(provider: NewsProvider):
    if provider.name in SOURCES:
        raise ValueError(f"Duplicate provider name: {provider.name}")
    SOURCES[provider.name] = provider


register_provider(ArchiveSource(
    Path(PROJECT["path_to_offline_demo_data"]),
    name="hn_snapshot", description="Saved Hacker News discussions for offline replay",
    chronological=True,
))
register_provider(HackerNewsSource(CONFIG["live"]))
JOURNAL = LoadJournal(DATA_DIR / "loads.jsonl")
app = FastAPI(title="Hacker News Collector", version="1.0.0")


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "service": "collector"}


@app.get("/sources")
def list_sources():
    return {"items": [
        {"id": provider.name, "description": provider.description} for provider in SOURCES.values()
        if not PROJECT["offline_demo"] or not provider.requires_network
    ]}


@app.get("/loads")
def recent_loads(limit: int = Query(default=20, ge=1, le=200)):
    try:
        return {"items": JOURNAL.recent(limit)}
    except OSError as exc:
        raise HTTPException(503, "Collector journal is temporarily unavailable") from exc


@app.post("/batches", response_model=BatchResponse)
def get_batch(request: BatchRequest):
    start = time.monotonic()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": request.source,
        "entity": request.entity,
        "limit": request.limit,
        "continued": request.cursor is not None,
    }
    try:
        provider = SOURCES.get(request.source)
        if provider is None:
            entry.update(status="error", error="Unknown provider")
            raise HTTPException(422, "Unknown provider. See GET /sources.")
        if PROJECT["offline_demo"] and provider.requires_network:
            entry.update(status="error", error="Network source is disabled in offline demo mode")
            raise HTTPException(409, "Network sources are disabled while offline_demo is enabled")
        news, next_cursor = provider.batch(
            aliases_for(request.entity, request.aliases), request.cursor, request.limit,
        )
        items = comments_as_dto(news)
        entry.update(status="ok", returned=len(items), has_more=next_cursor is not None)
        return {"items": items, "next_cursor": next_cursor, "source": request.source}
    except ValueError as exc:
        LOGGER.warning("Invalid Collector request: %s", exc)
        entry.update(status="error", error=str(exc))
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("Collector source failed")
        entry.update(status="error", error=str(exc))
        raise HTTPException(502, str(exc)) from exc
    finally:
        entry["duration_ms"] = round((time.monotonic() - start) * 1000, 1)
        JOURNAL.record(entry)
