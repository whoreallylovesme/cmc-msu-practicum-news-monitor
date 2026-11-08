"""Shared HTTP helpers for command-line clients of the monitor's public API."""

import base64
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def credentials(role):
    return (
        os.environ.get(f"MONITOR_{role.upper()}_USER", role),
        os.environ.get(f"MONITOR_{role.upper()}_PASSWORD", f"{role}-demo"),
    )


def call(base, path, *, query=None, body=None, method=None, role="admin", raw=False):
    headers = {"Accept": "application/json"}
    if role is not None:
        login = base64.b64encode(":".join(credentials(role)).encode()).decode()
        headers["Authorization"] = f"Basic {login}"
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    suffix = "?" + urlencode(query) if query else ""
    request = Request(base.rstrip("/") + path + suffix, data=data, headers=headers, method=method)
    with urlopen(request, timeout=float(os.environ.get("MONITOR_HTTP_TIMEOUT", "600"))) as response:
        return response.read().decode() if raw else json.load(response)


def wait_ready(base, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            services = call(base, "/api/status")["services"]
            if len(services) == 5 and all(
                row["status"] in {"ok", "healthy", "ready"} for row in services.values()
            ):
                return
        except URLError:
            pass
        time.sleep(0.75)
    raise RuntimeError("The five services did not become ready")


def monitoring_options(base, *, entity=None, source=None, teacher=None):
    config = call(base, "/api/config") if entity is None or source is None or teacher is None else {}
    monitoring = config.get("monitoring", {})
    selected_entity = monitoring["query"] if entity is None else entity
    selected_source = monitoring["source"] if source is None else source
    selected_teacher = config["pipeline"]["teacher"] if teacher is None else teacher
    return selected_entity, selected_source, selected_teacher


def train_model(base, entity, teacher, limit, timeout):
    model = call(base, "/services/ml/models", body={"entity": entity})
    model_id = model["id"]
    job = call(base, f"/services/ml/models/{model_id}/train", body={
        "teacher": teacher, "limit": limit, "mode": "from_scratch",
    })
    result = wait_job(base, job["job_id"], timeout)
    assert result["status"] == "succeeded", f"Expected a trained model: {result}"
    return call(base, f"/services/ml/models/{model_id}/summary")


def wait_job(base, job_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = call(base, f"/services/ml/jobs/{job_id}")
        if result["status"] in {"succeeded", "skipped"}:
            return result
        if result["status"] == "failed":
            raise RuntimeError(f'Training failed: {result.get("error") or result}')
        time.sleep(0.4)
    raise RuntimeError(f"Training did not finish: {job_id}")


def assert_status(base, expected, path, **kwargs):
    status = None
    try:
        call(base, path, **kwargs)
    except HTTPError as error:
        status = error.code
    assert status == expected, f"{path}: expected HTTP {expected}, received {status or 'success'}"


def all_items(base, path, params):
    offset = 0
    while True:
        page = call(base, path, query={**params, "limit": 100, "offset": offset})
        yield from page["items"]
        offset += len(page["items"])
        if not page["items"] or offset >= page["total"]:
            return
