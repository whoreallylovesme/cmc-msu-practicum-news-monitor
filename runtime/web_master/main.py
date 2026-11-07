import json
import logging
import math
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from lib.config import load_config, load_project_config
from lib.teacher_contracts import TeacherComment
from lib.contracts import ProviderName
from lib.sentiment import SCORE_MAX as MODEL_SCORE_MAX, SCORE_MIN as MODEL_SCORE_MIN


CONFIG = load_config(Path(__file__).with_name("config.yaml"), env_var="MASTER_CONFIG")
LOGGER = logging.getLogger(__name__)
TEMPLATES = Environment(
    loader=FileSystemLoader(Path(__file__).with_name("templates")),
    autoescape=select_autoescape(["html", "xml"]),
)
UPSTREAMS = {
    service: os.environ.get(f"{service.upper()}_URL", url).rstrip("/")
    for service, url in CONFIG["services"].items()
}
TEACHER_URL = os.environ.get("TEACHER_URL", CONFIG["teacher_url"]).rstrip("/")
SCHEMA_SERVICES = ("web_master", "collector", "storage", "ml", "visualization", "teacher")
CREDENTIALS = {
    role: (
        os.environ.get(f"MONITOR_{role.upper()}_USER", values["username"]),
        os.environ.get(f"MONITOR_{role.upper()}_PASSWORD", values["password"]),
    )
    for role, values in CONFIG["local_accounts"].items()
}
PROJECT = load_project_config()
MONITORING = {
    **PROJECT["monitoring"],
    "source": "hn_snapshot" if PROJECT["offline_demo"] else PROJECT["monitoring"]["source"],
}
SCORE_MIN = float(CONFIG["score_scale"]["min"])
SCORE_MAX = float(CONFIG["score_scale"]["max"])
if not math.isfinite(SCORE_MIN) or not math.isfinite(SCORE_MAX) or SCORE_MIN >= SCORE_MAX:
    raise ValueError("score_scale must have a finite minimum below its maximum")
SCORE_FACTOR = (SCORE_MAX - SCORE_MIN) / (MODEL_SCORE_MAX - MODEL_SCORE_MIN)
SCORE_SCALE = {"min": SCORE_MIN, "max": SCORE_MAX}

app = FastAPI(
    title="Hacker News Monitor API",
    description=(
        "The public API for the five-service monitor. Sign in with HTTP Basic. "
        "Readers can view reports and scores. Administrators can also import, train and score."
    ),
)
security = HTTPBasic(auto_error=False)
Bucket = Literal["day", "week"]


def role(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> str:
    if credentials is not None:
        for name, (username, password) in CREDENTIALS.items():
            correct_user = secrets.compare_digest(credentials.username.encode(), username.encode())
            correct_password = secrets.compare_digest(credentials.password.encode(), password.encode())
            if correct_user and correct_password:
                return name
    raise HTTPException(
        401, "Sign in with a monitor account", headers={"WWW-Authenticate": 'Basic realm="News Monitor"'}
    )


def admin(current_role: str = Depends(role)) -> None:
    if current_role != "admin":
        raise HTTPException(403, "Administrator access is required for this operation")


def upstream(service: str, method: str, path: str, *, data=None, params=None):
    query = {key: value for key, value in (params or {}).items() if value is not None}
    try:
        response = httpx.request(
            method,
            UPSTREAMS[service] + path,
            json=data,
            params=query,
            timeout=CONFIG["timeouts"].get(service, CONFIG["timeouts"]["default"]),
        )
    except httpx.TimeoutException as error:
        raise HTTPException(504, f"{service} timed out") from error
    except httpx.RequestError as error:
        raise HTTPException(502, f"{service} is unavailable") from error

    try:
        payload = response.json()
    except ValueError as error:
        raise HTTPException(502, f"{service} returned an invalid response") from error
    if response.is_error:
        detail = payload.get("detail", f"{service} request failed") if isinstance(payload, dict) else payload
        raise HTTPException(response.status_code, detail)
    return payload


def entity_settings(entity: str) -> dict:
    return {
        "name": entity,
        "aliases": MONITORING["aliases"] if entity == MONITORING["query"] else [],
        "matching": MONITORING["matching"],
    }


def versioned_model(model_id: str, entity: str) -> dict:
    model = upstream("ml", "GET", f"/models/{model_id}/summary")
    if model["entity"] != entity:
        raise HTTPException(409, "The model belongs to a different entity")
    if model.get("version") is None:
        raise HTTPException(409, "Train the model before requesting scores")
    return model


def prediction_key(model: dict) -> str:
    return f'{model["id"]}:{model["version"]}'


def display_scores(result: dict) -> dict:
    return {
        **result,
        "items": [
            {**item, "score": SCORE_MIN + (item["score"] - MODEL_SCORE_MIN) * SCORE_FACTOR}
            if item.get("score") is not None else item
            for item in result["items"]
        ],
    }


def display_metrics(value):
    if isinstance(value, dict):
        return {
            name: number * SCORE_FACTOR if name == "score_mae" and number is not None
            else display_metrics(number)
            for name, number in value.items()
        }
    if isinstance(value, list):
        return [display_metrics(item) for item in value]
    return value


class ImportRequest(BaseModel):
    entity: str = Field(min_length=1, max_length=120)
    source: ProviderName = "hn_snapshot"
    cursor: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=100)


class ModelRequest(BaseModel):
    entity: str = Field(min_length=1, max_length=120)
    kind: Literal["semantic", "softmax", "hard_label", "tiny_mlp"] = "hard_label"


class TrainRequest(BaseModel):
    teacher: Literal["dataset", "reviewed", "service", "snapshot", "fixture", "live"] = "dataset"
    mode: Literal["from_scratch", "incremental"] = "from_scratch"
    limit: int = Field(default=3000, ge=12, le=10000)


class ScoreRequest(BaseModel):
    entity: str = Field(min_length=1, max_length=120)
    limit: int = Field(default=500, ge=1, le=10000)


class ValidationRequest(BaseModel):
    items: list[dict] = Field(min_length=1, max_length=500)


class PredictionRequest(BaseModel):
    entity: str = Field(min_length=1, max_length=120)
    items: list[TeacherComment] = Field(min_length=1, max_length=100)
    version: Optional[int] = Field(default=None, ge=1)


class ImportResult(BaseModel):
    source: str
    received: int
    stored: int
    next_cursor: Optional[str]


class JobStarted(BaseModel):
    job_id: str


class ScoreResult(BaseModel):
    processed: int
    stored: int
    version: int


class Period(BaseModel):
    date: str
    score: Optional[float] = Field(description="Average on the configured display scale")
    count: int = Field(description="Number of scored comments judged relevant")
    total: int = Field(description="Number of scored comments, including off-topic comments")


class Timeline(BaseModel):
    items: list[Period]


@app.get("/health")
def health():
    return {"status": "ok", "service": "web_master"}


@app.get("/openapi/{service}.json", include_in_schema=False)
def service_schema(service: str):
    if service not in SCHEMA_SERVICES:
        raise HTTPException(404, "Unknown service")
    if service == "web_master":
        return app.openapi()
    base_url = TEACHER_URL if service == "teacher" else UPSTREAMS[service]
    try:
        response = httpx.get(base_url + "/openapi.json", timeout=CONFIG["timeouts"]["default"])
        response.raise_for_status()
        schema = response.json()
        if (not isinstance(schema, dict) or not isinstance(schema.get("openapi"), str)
                or not isinstance(schema.get("info"), dict) or not isinstance(schema.get("paths"), dict)):
            raise ValueError("Invalid OpenAPI schema")
    except (httpx.HTTPError, ValueError) as error:
        LOGGER.exception("Could not load the OpenAPI schema for %s", service)
        raise HTTPException(502, f"{service} API schema is unavailable") from error
    return schema


@app.get("/api/session", tags=["System"])
def session(current_role: str = Depends(role)):
    return {"role": current_role}


@app.get("/api/config", dependencies=[Depends(role)], tags=["System"])
def public_config():
    try:
        sources = upstream("collector", "GET", "/sources")["items"]
    except HTTPException:
        sources = []
    return {
        "entities": [entity_settings(MONITORING["query"])],
        "monitoring": MONITORING,
        "offline_demo": PROJECT["offline_demo"],
        "sources": sources,
        "score_scale": SCORE_SCALE,
    }


@app.get("/api/status", dependencies=[Depends(role)], tags=["System"])
def status():
    def probe(service):
        try:
            return service, upstream(service, "GET", "/health")
        except HTTPException as error:
            return service, {"status": "unavailable", "detail": error.detail}

    with ThreadPoolExecutor(max_workers=len(UPSTREAMS)) as executor:
        results = dict(executor.map(probe, UPSTREAMS))
    return {"services": {"web_master": health(), **results}}


@app.post("/api/import", dependencies=[Depends(admin)], response_model=ImportResult, tags=["Collection"])
def import_batch(request: ImportRequest):
    """Import one batch. Pass next_cursor back as cursor to continue the same collection."""
    settings = entity_settings(request.entity)
    batch = upstream(
        "collector",
        "POST",
        "/batches",
        data={**request.model_dump(), "aliases": settings.get("aliases", [])},
    )
    items = batch["items"]
    stored = {"stored": 0}
    if items:
        stored = upstream(
            "storage", "POST", "/comments/batch", data={"entity": request.entity, "items": items}
        )
    return {
        "source": batch["source"],
        "received": len(items),
        "stored": stored["stored"],
        "next_cursor": batch.get("next_cursor"),
    }


@app.get("/api/loads", dependencies=[Depends(role)], tags=["Collection"])
def load_history(limit: int = Query(default=15, ge=1, le=100)):
    return upstream("collector", "GET", "/loads", params={"limit": limit})


@app.get("/api/comments", dependencies=[Depends(role)], tags=["Collection"])
def comments(
    entity: str,
    query: Optional[str] = None,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Read or search stored comments using SQLite FTS5."""
    entity_settings(entity)
    return upstream(
        "storage", "GET", "/comments",
        params={"entity": entity, "query": query, "limit": limit, "offset": offset},
    )


@app.get("/api/models", dependencies=[Depends(role)], tags=["Models"])
def models(entity: str):
    entity_settings(entity)
    return display_metrics(upstream("ml", "GET", "/models", params={"entity": entity}))


@app.post("/api/models", status_code=201, dependencies=[Depends(admin)], tags=["Models"])
def create_model(request: ModelRequest):
    entity_settings(request.entity)
    return upstream("ml", "POST", "/models", data=request.model_dump())


@app.get("/api/models/{model_id}/summary", dependencies=[Depends(role)], tags=["Models"])
def model_summary(model_id: str):
    return {**display_metrics(upstream("ml", "GET", f"/models/{model_id}/summary")), "score_scale": SCORE_SCALE}


@app.post(
    "/api/models/{model_id}/train",
    status_code=202,
    dependencies=[Depends(admin)],
    response_model=JobStarted,
    tags=["Models"],
)
def train(model_id: str, request: TrainRequest):
    """Start training. Poll /api/jobs/{job_id} for its current stage and outcome."""
    if PROJECT["offline_demo"] and request.teacher == "live":
        raise HTTPException(409, "Offline demo uses saved labels. Select dataset, reviewed, snapshot or fixture.")
    return upstream("ml", "POST", f"/models/{model_id}/train", data=request.model_dump())


@app.get("/api/jobs/{job_id}", dependencies=[Depends(role)], tags=["Models"])
def job(job_id: str):
    return display_metrics(upstream("ml", "GET", f"/jobs/{job_id}"))


@app.post("/api/models/{model_id}/validate", dependencies=[Depends(admin)], tags=["Models"])
def validate(model_id: str, request: Optional[ValidationRequest] = None):
    result = upstream(
        "ml", "POST", f"/models/{model_id}/validate",
        data=request.model_dump() if request is not None else None,
    )
    return {**display_metrics(result), "score_scale": SCORE_SCALE}


@app.post("/api/models/{model_id}/predict", dependencies=[Depends(admin)], tags=["Models"])
def predict_items(model_id: str, request: PredictionRequest):
    """Apply a pinned model to new comments without importing them into Storage."""
    model = versioned_model(model_id, request.entity)
    result = upstream("ml", "POST", f"/models/{model_id}/predict", data={
        "items": [item.model_dump() for item in request.items],
        "version": request.version or model["version"],
    })
    return {**display_scores(result), "score_scale": SCORE_SCALE}


@app.post(
    "/api/models/{model_id}/score", dependencies=[Depends(admin)], response_model=ScoreResult, tags=["Models"]
)
def score(model_id: str, request: ScoreRequest):
    entity_settings(request.entity)
    model = versioned_model(model_id, request.entity)
    processed = 0
    stored = 0
    while processed < request.limit:
        page = upstream(
            "storage",
            "GET",
            "/comments",
            params={"entity": request.entity, "offset": processed, "limit": min(100, request.limit - processed)},
        )
        if not page["items"]:
            break
        predictions = upstream(
            "ml",
            "POST",
            f"/models/{model_id}/predict",
            data={"items": page["items"], "version": model["version"]},
        )
        result = upstream(
            "storage",
            "POST",
            "/predictions/batch",
            data={
                "entity": request.entity,
                "model_id": prediction_key(model),
                "items": predictions["items"],
            },
        )
        processed += len(page["items"])
        stored += result["stored"]
        if processed >= page["total"]:
            break
    return {"processed": processed, "stored": stored, "version": model["version"]}


@app.get("/api/timeline", dependencies=[Depends(role)], response_model=Timeline, tags=["Results"])
def timeline(entity: str, model_id: str, bucket: Bucket = "week"):
    """Read the active version's average score and sample size by UTC period."""
    model = versioned_model(model_id, entity)
    return display_scores(upstream(
        "storage",
        "GET",
        "/timeline",
        params={"entity": entity, "model_id": prediction_key(model), "bucket": bucket},
    ))


@app.get("/api/analytics", dependencies=[Depends(role)], tags=["Results"])
def analytics(entity: str, model_id: Optional[str] = None, bucket: Bucket = "week"):
    """Count mentions in collected comments and report their scored sentiment and sources."""
    settings = entity_settings(entity)
    model = versioned_model(model_id, entity) if model_id else None
    policy = model.get("mention_policy") if model else None
    aliases = policy["aliases"] if policy else settings["aliases"]
    matching = policy["options"] if policy else settings["matching"]
    result = upstream(
        "storage",
        "GET",
        "/analytics",
        params={
            "entity": entity,
            "model_id": prediction_key(model) if model else None,
            "bucket": bucket,
            "aliases": aliases,
            **matching,
        },
    )

    def scaled(counts):
        if counts["score"] is None:
            return counts
        return {**counts, "score": SCORE_MIN + (counts["score"] - MODEL_SCORE_MIN) * SCORE_FACTOR}

    return {
        "summary": scaled(result["summary"]),
        "periods": [scaled(period) for period in result["periods"]],
        "sources": [scaled(source) for source in result["sources"]],
        "score_scale": SCORE_SCALE,
    }


@app.get("/api/evidence", dependencies=[Depends(role)], tags=["Results"])
def evidence(
    entity: str,
    model_id: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Read scored comments. The optional UTC date range includes the start and excludes the end."""
    model = versioned_model(model_id, entity)
    return display_scores(upstream(
        "storage",
        "GET",
        "/predictions",
        params={
            "entity": entity,
            "model_id": prediction_key(model),
            "from_date": from_date,
            "to_date": to_date,
            "limit": limit,
            "offset": offset,
        },
    ))


@app.get("/api/report", dependencies=[Depends(role)], response_class=HTMLResponse, tags=["Results"])
def report(entity: str, model_id: str, bucket: Bucket = "week"):
    model = versioned_model(model_id, entity)
    data = display_scores(upstream(
        "storage", "GET", "/timeline",
        params={"entity": entity, "model_id": prediction_key(model), "bucket": bucket},
    ))
    page = TEMPLATES.get_template("report.html").render(
        entity=entity,
        model=model,
        teacher=model.get("teacher") or "—",
        points=data["items"],
        score_min=SCORE_MIN,
        score_max=SCORE_MAX,
        metrics=json.dumps(display_metrics(model.get("metrics") or {}), ensure_ascii=False, indent=2),
    )
    return HTMLResponse(page, headers={"Content-Disposition": 'attachment; filename="hn-monitor-report.html"'})
