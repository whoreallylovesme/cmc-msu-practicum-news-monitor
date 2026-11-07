import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, Response
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.concurrency import run_in_threadpool

from lib.config import load_config
from dashboard import create_dashboard
from runtime.visualization.openapi import SERVICES, merge_schemas


ROOT = Path(__file__).resolve().parent
CONFIG = load_config(ROOT / "config.yaml", env_var="VISUALIZATION_CONFIG")
LOGGER = logging.getLogger(__name__)
MASTER = os.environ.get("MASTER_URL", CONFIG["master_url"]).rstrip("/")
SCHEMA_SERVICES = tuple(service for service, _ in SERVICES)

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
def gateway_docs():
    return get_swagger_ui_html(
        openapi_url="/gateway-openapi.json",
        title="HN Monitor — all services",
        swagger_ui_parameters={"validatorUrl": None, "docExpansion": "none", "defaultModelsExpandDepth": -1},
    )


@app.get("/gateway-openapi.json", include_in_schema=False)
def gateway_schema(request: Request):
    schemas, unavailable = {}, []
    for service, title in SERVICES:
        try:
            result = gateway(request, "/openapi/" + service + ".json")
        except HTTPException:
            if service != "teacher":
                raise
            LOGGER.warning("Optional Teacher API schema is unavailable", exc_info=True)
            unavailable.append(title)
            continue
        if result.is_error:
            if service != "teacher":
                return forward(result)
            LOGGER.warning("Optional Teacher API schema returned HTTP %s", result.status_code)
            unavailable.append(title)
            continue
        schemas[service] = result.json()
    return JSONResponse(merge_schemas(schemas, unavailable), headers={"cache-control": "no-store"})


@app.api_route("/services/{service}/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE"],
               include_in_schema=False)
async def service_request(service: str, path: str, request: Request):
    if service not in SCHEMA_SERVICES:
        raise HTTPException(404, "Unknown service")
    body = await request.body()
    result = await run_in_threadpool(gateway, request, "/services/" + service + "/" + path, body)
    return forward(result)


dashboard = create_dashboard(MASTER, CONFIG["timeout_seconds"])
app.mount("/", WSGIMiddleware(dashboard.server))
