"""ML service for training, validating and serving versioned sentiment models."""

from itertools import islice
import hashlib
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional
from uuid import uuid4

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from lib.config import load_config, load_monitoring, load_project_config
from lib.cli import configure_logging
from lib.teacher_contracts import input_hash as service_input_hash
from lib.supervision import load_auxiliary

import student
import training_data as selection
from dataset_labels import fingerprint, load_dataset_snapshot, supervision_hash
from registry import Registry, serialize
from teacher_client import TeacherService


configure_logging()
LOGGER = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent
CONFIG = load_config(
    BASE / "config.yaml",
    env_var="ML_CONFIG",
    overrides={
        "storage_url": "STORAGE_URL",
        "teacher_service_url": "TEACHER_SERVICE_URL",
    },
)
CONFIG["monitoring"] = load_monitoring()
PIPELINE = load_project_config()["pipeline"]

DATA = Path(os.environ.get("MONITOR_DATA_DIR", CONFIG["data_dir"]))
DATA.mkdir(parents=True, exist_ok=True)
REGISTRY = Registry(DATA / "model-registry.sqlite3")
LOCK = threading.RLock()
WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")
STATE = REGISTRY.state()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def persist():
    REGISTRY.save(STATE)


for entry in STATE["jobs"].values():
    if entry["status"] in {"queued", "running"}:
        entry.update(status="failed", error="Training interrupted by service restart", finished_at=timestamp())
        model = STATE["models"][entry["model_id"]]
        model["status"] = "ready" if model["version"] else "failed"
        model["last_error"] = entry["error"]
persist()


class Comment(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    story_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    story_title: str = ""
    parent_text: Optional[str] = None
    parent_id: Optional[str] = Field(default=None, max_length=200)
    content_hash: Optional[str] = None
    human_label: Optional[int] = Field(default=None, ge=0, le=5)


class CreateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity: str = Field(min_length=1, max_length=200)


class TrainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    teacher: Literal["dataset", "service"] = PIPELINE["teacher"]
    mode: Literal["from_scratch", "incremental"] = "from_scratch"
    limit: int = Field(default=PIPELINE["training_limit"], ge=4, le=10000)


class Inference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[Comment] = Field(min_length=1, max_length=5000)
    version: Optional[int] = Field(default=None, ge=1)


class Validation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: Optional[List[Comment]] = Field(default=None, max_length=5000)
    version: Optional[int] = Field(default=None, ge=1)


app = FastAPI(title="News Monitor ML Service", version="1.0.0")


def storage(session, path, params=None, data=None):
    url = f"{CONFIG['storage_url'].rstrip('/')}{path}"
    response = session.get(url, params=params, timeout=CONFIG["storage_timeout"]) if data is None else (
        session.post(url, json=data, timeout=CONFIG["storage_timeout"])
    )
    if not response.ok:
        detail = response.text[:500]
        raise RuntimeError(f"Storage {path} returned HTTP {response.status_code}: {detail}")
    return response.json()


def pages(session, path, params, limit=None):
    count = 0
    page_size = int(CONFIG["storage_page_size"])
    while limit is None or count < limit:
        requested = page_size if limit is None else min(page_size, limit - count)
        data = storage(session, path, {**params, "offset": count, "limit": requested})
        items = data["items"]
        yield from items
        count += len(items)
        if not items or count >= data["total"]:
            break


def model_locked(model_id):
    model = STATE["models"].get(model_id)
    if model is None:
        raise HTTPException(404, "Model not found")
    return model


def load_artifact(model_id, version):
    with LOCK:
        model = model_locked(model_id)
        selected = model["version"] if version is None else version
        if selected is None:
            raise HTTPException(409, "Model has no trained version")
        if not any(entry["version"] == selected for entry in model["versions"]):
            raise HTTPException(404, "Model version not found")
    try:
        return REGISTRY.load(model_id, selected), selected
    except (OSError, ValueError, sqlite3.Error) as error:
        LOGGER.exception("Could not load model %s v%s", model_id, selected)
        raise HTTPException(503, "Model artifact is unavailable") from error


def job_progress(job_id, **updates):
    with LOCK:
        STATE["jobs"][job_id].update(updates)
        persist()


def entity_aliases(entity):
    monitoring = CONFIG["monitoring"]
    aliases = monitoring["aliases"] if entity.casefold() == monitoring["query"].casefold() else []
    return aliases


def labels_from_service(session, job_id, entity, items, service):
    aliases = entity_aliases(entity)
    hashes = {item["id"]: service_input_hash(entity, item, aliases) for item in items}
    stored = {
        row["comment_id"]: student.normalized_distribution(row["probabilities"]).tolist()
        for row in pages(session, "/labels", {"entity": entity, "teacher": service.namespace})
        if hashes.get(row["comment_id"]) == row.get("input_hash") and row["comment_id"] in hashes
    }
    cached = len(stored)
    missing = [item for item in items if item["id"] not in stored]
    size = int(CONFIG.get("teacher_service_batch_size", 16))
    if not 1 <= size <= 100:
        raise ValueError("teacher_service_batch_size must be between 1 and 100")
    started = time.perf_counter()
    for offset in range(0, len(missing), size):
        batch = missing[offset:offset + size]
        labels = service.label(entity, batch, aliases)
        by_id = {item["id"]: item for item in batch}
        records = []
        for label in labels:
            record = label.model_dump()
            item = by_id[label.comment_id]
            if item.get("content_hash"):
                record["content_hash"] = item["content_hash"]
            records.append(record)
            stored[label.comment_id] = label.probabilities
        storage(session, "/labels/batch", data={"entity": entity, "teacher": service.namespace, "items": records})
        job_progress(job_id, stage="labeling", labeled=len(stored), total=len(items))
    return [stored[item["id"]] for item in items], service.namespace, {
        "labels_from_cache": cached, "teacher_inference_items": len(missing),
        "teacher_ms_per_item": (time.perf_counter() - started) * 1000 / len(missing) if missing else None,
        "service_teacher": service.metadata,
    }


def labels_for(session, job_id, teacher, entity, items, snapshot, service=None):
    if teacher == "service":
        return labels_from_service(session, job_id, entity, items, service)
    namespace = snapshot["namespace"]
    hashes = {item["id"]: fingerprint(item) for item in items}
    stored = {
        row["comment_id"]: student.normalized_distribution(row["probabilities"]).tolist()
        for row in pages(session, "/labels", {"entity": entity, "teacher": namespace})
        if hashes.get(row["comment_id"]) == row.get("input_hash") and row["comment_id"] in hashes
    }
    cached = len(stored)
    missing = [item for item in items if item["id"] not in stored]
    for offset in range(0, len(missing), 100):
        records = []
        for item in missing[offset:offset + 100]:
            target = snapshot["labels"][item["id"]]["probabilities"]
            stored[item["id"]] = target
            record = {"comment_id": item["id"], "probabilities": target, "input_hash": hashes[item["id"]]}
            if item.get("content_hash"):
                record["content_hash"] = item["content_hash"]
            records.append(record)
        storage(session, "/labels/batch", data={"entity": entity, "teacher": namespace, "items": records})
        job_progress(job_id, stage="labeling", labeled=len(stored), total=len(items))
    return [stored[item["id"]] for item in items], namespace, {
        "labels_from_cache": cached, "teacher_inference_items": 0,
        "teacher_ms_per_item": None, "labels_from_dataset": len(items),
        "dataset_source": snapshot["teacher"],
    }


def teacher_snapshot(teacher, entity):
    if teacher == "dataset":
        return load_dataset_snapshot(load_project_config()["path_to_offline_demo_data"], entity,
                                     aliases=entity_aliases(entity))
    return None


def eligible_comments(session, teacher, entity, snapshot, service=None):
    aliases = entity_aliases(entity)
    for item in pages(session, "/comments", {"entity": entity}):
        supervision = []
        if snapshot:
            record = snapshot["labels"].get(item["id"])
            if not record:
                continue
            if record["fingerprint"] != fingerprint(item):
                raise ValueError(f"Stale {snapshot['kind']} label for comment {item['id']}: raw context changed")
            supervision.append(supervision_hash(record))
        if service:
            supervision.append(service.revision + service_input_hash(entity, item, aliases))
        if supervision:
            identity = supervision[0] if len(supervision) == 1 else hashlib.sha256(":".join(supervision).encode()).hexdigest()
            item = {**item, "supervision_hash": identity}
        yield item


def train_job(job_id):
    with LOCK:
        job = STATE["jobs"][job_id]
        model_id, teacher, limit = job["model_id"], job["teacher"], job["limit"]
        mode = job.get("mode", "from_scratch")
        model = STATE["models"][model_id]
        entity, kind = model["entity"], model["kind"]
        base_version = model["version"]
        version = (base_version or 0) + 1
    try:
        job_progress(job_id, status="running", stage="fetching", started_at=timestamp())
        prior = REGISTRY.load(model_id, base_version) if mode == "incremental" else None
        if prior is not None and "training_history" not in prior:
            raise ValueError("This revision predates incremental history. Train once in from_scratch mode.")
        auxiliary, auxiliary_targets, auxiliary_metadata = load_auxiliary(
            CONFIG, mode=mode, version=version, seed=CONFIG["student_encoder"].get("seed", 42),
        )
        if prior is not None and prior.get("auxiliary_supervision", {}).get("sha256") != auxiliary_metadata.get("sha256"):
            raise ValueError("Auxiliary supervision changed; train from scratch")

        with requests.Session() as session:
            service = TeacherService(CONFIG, session) if teacher == "service" else None
            snapshot = teacher_snapshot(teacher, entity)
            items = eligible_comments(session, teacher, entity, snapshot, service)
            if prior is not None:
                plan = selection.incremental(items, prior["training_history"], limit, CONFIG)
                if not plan["train"]:
                    with LOCK:
                        model.update(status="ready", updated_at=timestamp(), last_error=None)
                        job.update(
                            status="skipped", stage="complete", version=base_version,
                            reason="No new or edited comments in eligible training threads",
                            finished_at=timestamp(), **plan["counts"],
                        )
                        persist()
                    return
                if not plan["holdout"]:
                    raise ValueError("Incremental training needs at least one available held-out comment")
            else:
                sample = list(islice(items, limit))
                if len(sample) < 4 and snapshot:
                    raise ValueError("Import the matching Hacker News snapshot before training")
                plan = selection.from_scratch(sample, limit, float(CONFIG["validation_fraction"]))

            train, holdout = plan["train"], plan["holdout"]
            targets, teacher_revision, timing = labels_for(
                session, job_id, teacher, entity, train + holdout, snapshot, service,
            )

        job_progress(job_id, stage="fitting", **plan["counts"])
        artifact, optimization = student.fit(
            train + auxiliary, targets[:len(train)] + auxiliary_targets, kind, {
                **CONFIG, "entity": entity,
                "aliases": entity_aliases(entity),
            }, prior=prior,
        )
        artifact["training_history"] = plan["history"]
        artifact["auxiliary_supervision"] = auxiliary_metadata
        report = student.evaluate(
            artifact, holdout, targets[len(train):],
        )
        report["labelled_items"] = len(holdout)
        metrics = {
            "mode": mode,
            "base_version": base_version if prior is not None else None,
            "training_items": len(train),
            "auxiliary_training_items": len(auxiliary),
            "total_training_examples": len(train) + len(auxiliary),
            "training_threads": len({item["story_id"] for item in train}),
            "holdout": report,
            "optimization": optimization,
            **plan["counts"],
            **timing,
        }

        payload, checksum = serialize(artifact)
        record = {
            "version": version,
            "teacher": teacher,
            "teacher_revision": teacher_revision,
            "mode": mode,
            "base_version": base_version if prior is not None else None,
            "trained_at": timestamp(),
            "metrics": metrics,
            "training_story_ids": sorted(story for story, role in plan["history"]["roles"].items() if role == "train"),
            "holdout_story_ids": sorted(story for story, role in plan["history"]["roles"].items() if role == "holdout"),
            "artifact_bytes": len(payload),
            "artifact_sha256": checksum,
            "score_statistic": artifact.get("score_statistic", "mean"),
        }
        with LOCK:
            updated_model = {
                **model,
                "version": version,
                "status": "ready",
                "teacher": teacher,
                "mode": mode,
                "metrics": metrics,
                "updated_at": timestamp(),
                "last_error": None,
                "score_statistic": artifact.get("score_statistic", "mean"),
                "versions": [*model["versions"], record],
            }
            updated_job = {
                **job,
                "status": "succeeded",
                "stage": "complete",
                "version": version,
                "metrics": metrics,
                "finished_at": timestamp(),
            }
            new_state = {
                "models": {**STATE["models"], model_id: updated_model},
                "jobs": {**STATE["jobs"], job_id: updated_job},
            }
            REGISTRY.save(new_state, (model_id, version, checksum, payload))
            STATE.update(new_state)
    except Exception as error:
        LOGGER.exception("Training job %s failed", job_id)
        with LOCK:
            model["status"] = "ready" if model["version"] else "failed"
            model["last_error"] = str(error)
            model["updated_at"] = timestamp()
            job.update(status="failed", error=str(error), finished_at=timestamp())
            persist()


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "service": "ml", "default_training_source": PIPELINE["teacher"],
            "teacher_service_url_configured": bool(CONFIG.get("teacher_service_url"))}


@app.post("/models", status_code=201)
def create_model(payload: CreateModel):
    entity = payload.entity.strip()
    if not entity:
        raise HTTPException(422, "Entity cannot be blank")
    model = {
        "id": uuid4().hex,
        "entity": entity,
        "kind": "encoder",
        "status": "new",
        "version": None,
        "teacher": None,
        "metrics": None,
        "created_at": timestamp(),
        "updated_at": timestamp(),
        "last_error": None,
        "versions": [],
    }
    with LOCK:
        STATE["models"][model["id"]] = model
        persist()
        return dict(model)


@app.get("/models")
def list_models(entity: Optional[str] = None):
    with LOCK:
        models = [
            {key: value for key, value in model.items() if key != "versions"}
            for model in STATE["models"].values()
            if entity is None or entity.casefold() == model["entity"].casefold()
        ]
        return {"items": sorted(models, key=lambda model: model["created_at"], reverse=True)}


@app.get("/models/{model_id}/summary")
def summary(model_id: str):
    with LOCK:
        model = model_locked(model_id)
        return {**model, "versions": list(model["versions"])}


@app.post("/models/{model_id}/train", status_code=202)
def start_training(model_id: str, payload: TrainModel):
    with LOCK:
        model = model_locked(model_id)
        if model["status"] == "training":
            raise HTTPException(409, "This model already has a training job")
        if payload.mode == "incremental" and model["version"] is None:
            raise HTTPException(409, "Train from scratch before starting incremental training")
        if payload.mode == "incremental" and model["teacher"] != payload.teacher:
            raise HTTPException(409, "Teacher changed; train from scratch before switching label sources")
        job_id = uuid4().hex
        job = {
            "id": job_id,
            "model_id": model_id,
            "teacher": payload.teacher,
            "mode": payload.mode,
            "limit": payload.limit,
            "status": "queued",
            "stage": "queued",
            "created_at": timestamp(),
        }
        STATE["jobs"][job_id] = job
        model.update(status="training", updated_at=timestamp(), last_error=None)
        persist()
        WORKER.submit(train_job, job_id)
        return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with LOCK:
        job = STATE["jobs"].get(job_id)
        if job is None:
            raise HTTPException(404, "Training job not found")
        return dict(job)


@app.post("/models/{model_id}/predict")
def predict(model_id: str, payload: Inference):
    artifact, version = load_artifact(model_id, payload.version)
    return {
        "model_id": model_id,
        "version": version,
        "items": student.predictions(artifact, [item.model_dump() for item in payload.items]),
    }


@app.post("/models/{model_id}/validate")
def validate(model_id: str, payload: Optional[Validation] = None):
    artifact, version = load_artifact(model_id, payload.version if payload else None)
    if payload is None or payload.items is None:
        with LOCK:
            record = next(entry for entry in model_locked(model_id)["versions"] if entry["version"] == version)
        return {"model_id": model_id, "version": version, "metrics": record["metrics"]["holdout"]}
    items = [item.model_dump() for item in payload.items]
    if not items or not any(item["human_label"] is not None for item in items):
        raise HTTPException(422, "Validation items require human_label")
    return {"model_id": model_id, "version": version, "metrics": student.evaluate(artifact, items)}
