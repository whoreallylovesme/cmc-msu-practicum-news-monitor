import json
import os
import re
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, Response
from starlette.middleware.wsgi import WSGIMiddleware

from lib.config import load_config
from dashboard import create_dashboard


ROOT = Path(__file__).resolve().parent
CONFIG = load_config(ROOT / "config.yaml", env_var="VISUALIZATION_CONFIG")
MASTER = os.environ.get("MASTER_URL", CONFIG["master_url"]).rstrip("/")
SCHEMA_SERVICES = ("web_master", "collector", "storage", "ml", "visualization", "teacher")
GET_ROUTES = re.compile(
    r"(session|config|status|loads|comments|models|timeline|analytics|evidence|report"
    r"|models/[A-Za-z0-9_-]+/summary|jobs/[A-Za-z0-9_-]+)"
)
POST_ROUTES = re.compile(r"(import|models|models/[A-Za-z0-9_-]+/(train|validate|predict|score))")

app = FastAPI(title="News Monitor Visualization")


def gateway(request: Request, path: str, body=None):
    headers = {
        name: request.headers[name]
        for name in ("authorization", "content-type")
        if name in request.headers
    }
    try:
        return httpx.request(
            request.method,
            f"{MASTER}{path}",
            params=list(request.query_params.multi_items()),
            headers=headers,
            content=body,
            timeout=CONFIG["timeout_seconds"],
        )
    except httpx.TimeoutException as error:
        raise HTTPException(504, "Web Master did not reply in time") from error
    except httpx.RequestError as error:
        raise HTTPException(502, "Web Master is unavailable") from error


def forward(result):
    headers = {
        name: result.headers[name]
        for name in ("content-type", "content-disposition", "www-authenticate")
        if name in result.headers
    }
    headers["cache-control"] = "no-store"
    return Response(content=result.content, status_code=result.status_code, headers=headers)


@app.get("/health")
def health():
    return {"status": "ok", "service": "visualization"}


@app.get("/gateway-docs", include_in_schema=False)
def gateway_docs(service: str = "web_master"):
    if service not in SCHEMA_SERVICES:
        raise HTTPException(404, "Unknown service")
    parameters = {"validatorUrl": None}
    if service != "web_master":
        parameters["supportedSubmitMethods"] = []
    return get_swagger_ui_html(
        openapi_url="/gateway-openapi.json?service=" + service,
        title="HN Monitor API: " + service,
        swagger_ui_parameters=parameters,
    )


@app.get("/gateway-openapi.json", include_in_schema=False)
def gateway_schema(request: Request, service: str = "web_master"):
    if service not in SCHEMA_SERVICES:
        raise HTTPException(404, "Unknown service")
    result = gateway(request, "/openapi/" + service + ".json")
    if result.is_error:
        return forward(result)
    schema = result.json()
    info = schema["info"]
    navigation = "API schemas: " + " · ".join(
        f"[{name}](/gateway-docs?service={name})" for name in SCHEMA_SERVICES
    )
    description = info.get("description", "")
    if service != "web_master":
        description += "\n\nInternal service API, read-only in Swagger; request execution is disabled."
    info["description"] = (description + "\n\n" + navigation).strip()
    return JSONResponse(schema, headers={"cache-control": "no-store"})


@app.get("/api/{path:path}")
def view_data(request: Request, path: str):
    if not GET_ROUTES.fullmatch(path):
        raise HTTPException(404, "Unknown dashboard operation")
    return forward(gateway(request, "/api/" + path))


@app.post("/api/{path:path}")
def change_data(request: Request, path: str, body: dict = None):
    if not POST_ROUTES.fullmatch(path):
        raise HTTPException(404, "Unknown dashboard operation")
    content = json.dumps(body).encode() if body is not None else None
    return forward(gateway(request, "/api/" + path, content))


dashboard = create_dashboard(MASTER, CONFIG["timeout_seconds"])
app.mount("/", WSGIMiddleware(dashboard.server))
