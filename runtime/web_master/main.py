import json
import logging
import math
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from lib.config import load_config, load_project_config
from lib.contracts import Identifier, ProviderName
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
    entity: Identifier
    source: ProviderName = "hn_snapshot"
    cursor: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=100)


class ScoreRequest(BaseModel):
    model_id: Identifier
    entity: Identifier
    limit: int = Field(default=500, ge=1, le=10000)


class ImportResult(BaseModel):
    source: str
    received: int
    stored: int
    next_cursor: Optional[str]


class ScoreResult(BaseModel):
    processed: int
    stored: int
    version: int


@app.get("/health", include_in_schema=False)
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


@app.api_route("/services/{service}/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE"],
               include_in_schema=False)
async def service_request(service: str, path: str, request: Request, current_role: str = Depends(role)):
    if service not in SCHEMA_SERVICES:
        raise HTTPException(404, "Unknown service")
    if any(part in {".", ".."} for part in path.split("/")):
        raise HTTPException(404, "Unknown service operation")
    read_model = service == "ml" and request.method == "GET"
    if service != "web_master" and not read_model and current_role != "admin":
        raise HTTPException(403, "Administrator access is required for internal service requests")
    if service == "web_master":
        if not (path.startswith("api/") or path == "health"):
            raise HTTPException(404, "Unknown Web Master operation")
        base_url = "http://web-master"
    else:
        base_url = TEACHER_URL if service == "teacher" else UPSTREAMS[service]
    headers = {name: request.headers[name] for name in ("authorization", "content-type", "accept")
               if name in request.headers}
    body = await request.body()
    try:
        options = {"params": list(request.query_params.multi_items()), "headers": headers, "content": body}
        if service == "web_master":
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url) as client:
                response = await client.request(request.method, "/" + path, **options)
        else:
            response = await run_in_threadpool(
                httpx.request, request.method, base_url + "/" + path, **options,
                timeout=CONFIG["timeouts"].get(service, CONFIG["timeouts"]["default"]),
            )
    except httpx.TimeoutException as error:
        LOGGER.exception("Service request timed out for %s", service)
        raise HTTPException(504, f"{service} timed out") from error
    except httpx.RequestError as error:
        LOGGER.exception("Service request failed for %s", service)
        raise HTTPException(502, f"{service} is unavailable") from error
    outgoing = {name: response.headers[name]
                for name in ("content-type", "content-disposition", "www-authenticate")
                if name in response.headers}
    model_response = path == "models" or path.startswith(("models/", "jobs/"))
    if service == "ml" and model_response and not response.is_error:
        payload = display_metrics(response.json())
        if path.endswith("/predict"):
            payload = display_scores(payload)
        return JSONResponse({**payload, "score_scale": SCORE_SCALE}, status_code=response.status_code)
    return Response(content=response.content, status_code=response.status_code, headers=outgoing)


@app.get("/api/config", tags=["System"])
def public_config(current_role: str = Depends(role)):
    try:
        sources = upstream("collector", "GET", "/sources")["items"]
    except HTTPException:
        sources = []
    return {
        "role": current_role,
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
    try:
        loads = upstream("collector", "GET", "/loads", params={"limit": 5})["items"]
    except HTTPException:
        loads = []
    return {"services": {"web_master": health(), **results}, "loads": loads}


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


@app.get("/api/comments", dependencies=[Depends(role)], tags=["Collection"])
def comments(
    entity: Identifier,
    model_id: Optional[str] = None,
    query: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Read stored comments. Set model_id for scored comments and optional UTC dates.

    Without model_id, query searches the raw archive using SQLite FTS5.
    """
    if model_id:
        model = versioned_model(model_id, entity)
        return display_scores(upstream(
            "storage", "GET", "/predictions",
            params={
                "entity": entity, "model_id": prediction_key(model),
                "query": query, "from_date": from_date, "to_date": to_date, "limit": limit, "offset": offset,
            },
        ))
    return upstream(
        "storage", "GET", "/comments",
        params={"entity": entity, "query": query, "from_date": from_date, "to_date": to_date,
                "limit": limit, "offset": offset},
    )


@app.post(
    "/api/score", dependencies=[Depends(admin)], response_model=ScoreResult, tags=["Results"]
)
def score(request: ScoreRequest):
    """Score stored comments with a pinned model version and save the results."""
    model_id = request.model_id
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


def model_analytics(entity, model, bucket):
    settings = entity_settings(entity)
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


@app.get("/api/analytics", dependencies=[Depends(role)], tags=["Results"])
def analytics(entity: Identifier, model_id: Optional[str] = None, bucket: Bucket = "week"):
    """Count mentions in collected comments and report their scored sentiment and sources."""
    model = versioned_model(model_id, entity) if model_id else None
    return model_analytics(entity, model, bucket)


@app.get("/api/report", dependencies=[Depends(role)], response_class=HTMLResponse, tags=["Results"])
def report(entity: Identifier, model_id: str, bucket: Bucket = "week"):
    model = versioned_model(model_id, entity)
    data = model_analytics(entity, model, bucket)
    page = TEMPLATES.get_template("report.html").render(
        entity=entity,
        model=model,
        teacher=model.get("teacher") or "—",
        points=[{**point, "count": point["relevant"], "total": point["scored"]}
                for point in data["periods"]],
        score_min=SCORE_MIN,
        score_max=SCORE_MAX,
        metrics=json.dumps(display_metrics(model.get("metrics") or {}), ensure_ascii=False, indent=2),
    )
    return HTMLResponse(page, headers={"Content-Disposition": 'attachment; filename="hn-monitor-report.html"'})
