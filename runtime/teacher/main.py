"""HTTP service for a lazily loaded, target-conditioned GGUF teacher."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
for directory in (ROOT, BASE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fastapi import FastAPI, HTTPException

from lib.teacher_contracts import TeacherRequest, TeacherResponse
from lib.cli import configure_logging
from model import TeacherEngine, load_settings, prepare_assets


configure_logging()
LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    from lib.cli import load_cli_config

    config_path, _ = load_cli_config(BASE / "config.yaml")
    os.environ["TEACHER_CONFIG"] = str(config_path)

SETTINGS = load_settings()
ENGINE = TeacherEngine(SETTINGS)


@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        ENGINE.close()


app = FastAPI(title="Entity Sentiment Teacher", version="2.0.0", redoc_url=None,
              lifespan=lifespan, swagger_ui_parameters={"validatorUrl": None})


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "service": "teacher", "revision": SETTINGS["revision"],
            "model_loaded": ENGINE.runtime is not None}


@app.get("/metadata")
def metadata():
    return ENGINE.metadata()


@app.post("/label", response_model=TeacherResponse)
def label(request: TeacherRequest):
    try:
        return ENGINE.label(request)
    except Exception as error:
        LOGGER.exception("The configured GGUF teacher could not produce labels")
        raise HTTPException(503, "Teacher model is unavailable; check its configuration and logs") from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SETTINGS["host"], port=SETTINGS["port"])
