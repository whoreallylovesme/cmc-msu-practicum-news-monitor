"""Independent CPU ONNX teacher service; no student or autoregressive runtime."""

import hashlib
import json
import logging
import os
import string
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException

from lib.mentions import MentionMatcher
from lib.model_assets import asset_spec, ensure_asset
from lib.sentiment import CHOICES
from lib.teacher_contracts import INPUT_HASH_VERSION, TeacherRequest, TeacherResponse, input_hash


LOGGER = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent
TEMPLATE_FIELDS = {"entity", "text", "story_title", "parent_text", "title", "parent"}


def _template(value, required=()):
    if not isinstance(value, str) or not value:
        raise ValueError("Teacher input templates must be nonempty strings")
    fields = set()
    for _, name, spec, conversion in string.Formatter().parse(value):
        if name is not None:
            if name not in TEMPLATE_FIELDS or spec or conversion:
                raise ValueError("Teacher templates support only simple named text/context fields")
            fields.add(name)
    if not set(required) <= fields:
        raise ValueError(f"Teacher template must contain {', '.join(required)}")
    return value


def load_settings(path=None):
    selected = Path(path or os.environ.get("TEACHER_CONFIG", BASE / "config.yaml")).expanduser().resolve()
    config = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Teacher config must be a YAML mapping")
    backend = config.get("backend", "sequence_classifier")
    if backend not in {"sequence_classifier", "nli"}:
        raise ValueError("Teacher backend must be sequence_classifier or nli")
    model = asset_spec(config.get("model_weights"), size_bytes=config.get("model_size_bytes"),
                       max_bytes=int(config.get("max_model_bytes", 1024 * 1024 * 1024)), base_dir=selected.parent)
    tokenizer = asset_spec(config.get("tokenizer"), size_bytes=config.get("tokenizer_size_bytes"),
                           max_bytes=32 * 1024 * 1024, base_dir=selected.parent)
    length, threads = config.get("max_length", 512), config.get("threads", 2)
    padding_id, padding_token = config.get("pad_token_id", 0), config.get("pad_token", "[PAD]")
    if (type(length) is not int or not 8 <= length <= 4096 or type(threads) is not int or threads < 1
            or type(padding_id) is not int or padding_id < 0
            or not isinstance(padding_token, str) or not padding_token):
        raise ValueError("Invalid teacher token limit, CPU threads or padding configuration")
    mapping = config.get("class_mapping", ["B", "C", "D"])
    if not isinstance(mapping, list) or not mapping or any(value not in CHOICES for value in mapping):
        raise ValueError("class_mapping must map each source class index to A, B, C, D, E or F")
    sequence = dict(config.get("sequence") or {})
    nli = dict(config.get("nli") or {})
    if backend == "sequence_classifier":
        input_format = sequence.get("input_format", "pair")
        if input_format not in {"pair", "text"}:
            raise ValueError("Sequence teacher input_format must be pair or text")
        text_template = _template(sequence.get("text_template", "{text}"), ["text"])
        target_template = _template(sequence.get("target_template", "{entity}"), ["entity"])
        if input_format == "text" and "{entity}" not in text_template:
            raise ValueError("Single-text teacher template must include {entity}")
        adapter = {"input_format": input_format, "text_template": text_template,
                   "target_template": target_template if input_format == "pair" else None}
    else:
        hypotheses = nli.get("hypotheses")
        entailment = nli.get("entailment_index", 0)
        if (not isinstance(hypotheses, list) or len(hypotheses) != 3
                or type(entailment) is not int or entailment < 0 or len(mapping) != 3):
            raise ValueError("NLI requires three hypotheses, three class mappings and an entailment index")
        adapter = {"premise_template": _template(nli.get("premise_template", "{text}"), ["text"]),
                   "hypotheses": [_template(value, ["entity"]) for value in hypotheses],
                   "entailment_index": entailment}
    cache = Path(os.environ.get("TEACHER_ENCODER_CACHE_DIR") or config.get("cache_dir", ".local-data/teacher-models")).expanduser()
    if not cache.is_absolute():
        cache = ROOT / cache
    protocol = {"version": "onnx-entity-teacher-v1", "backend": backend, "adapter": adapter,
                "class_mapping": mapping, "max_length": length,
                "padding": {"id": padding_id, "token": padding_token},
                "input_hash_version": INPUT_HASH_VERSION,
                "mention_policy": "request-scoped MentionMatcher; outside scope is class F",
                "distribution_kind": "hypothesis_scores_softmax" if backend == "nli" else "model_softmax",
                "calibrated": False}
    identity = {"protocol": protocol, "model_sha256": model.sha256, "tokenizer_sha256": tokenizer.sha256}
    revision = "hf-" + hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                                 separators=(",", ":")).encode()).hexdigest()
    return {"backend": backend, "model": model, "tokenizer": tokenizer, "protocol": protocol,
            "revision": revision, "threads": threads, "cache_dir": cache,
            "download_timeout": float(config.get("download_timeout", 180)),
            "model_name": config.get("model_name"), "model_revision": config.get("model_revision")}


def prepare_assets(settings, cache_dir=None):
    selected = Path(cache_dir or settings["cache_dir"])
    return {name: ensure_asset(settings[name], selected, suffix=suffix,
                               timeout=settings["download_timeout"])
            for name, suffix in (("model", ".onnx"), ("tokenizer", ".json"))}


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Teacher returned invalid class logits")
    probabilities = np.exp(values - values.max())
    return probabilities / probabilities.sum()


def _fields(entity, item):
    return {"entity": entity, "text": item.text, "story_title": item.story_title,
            "title": item.story_title, "parent_text": item.parent_text or "", "parent": item.parent_text or ""}


class TeacherRuntime:
    def __init__(self, settings):
        import onnxruntime
        from tokenizers import Tokenizer

        paths = prepare_assets(settings)
        self.settings = settings
        self.tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
        protocol = settings["protocol"]
        self.tokenizer.enable_truncation(max_length=protocol["max_length"], strategy="only_first")
        self.tokenizer.enable_padding(pad_id=protocol["padding"]["id"], pad_token=protocol["padding"]["token"])
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = settings["threads"]
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(paths["model"]), sess_options=options,
                                                   providers=["CPUExecutionProvider"])
        self.inputs = {item.name for item in self.session.get_inputs()}
        if not self.inputs <= {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("Configured teacher model requires unsupported ONNX inputs")

    def distribution(self, entity, item):
        protocol = self.settings["protocol"]
        adapter = protocol["adapter"]
        fields = _fields(entity, item)
        if self.settings["backend"] == "nli":
            premise = adapter["premise_template"].format_map(fields)
            texts = [(premise, template.format_map(fields)) for template in adapter["hypotheses"]]
        else:
            text = adapter["text_template"].format_map(fields)
            texts = [(text, adapter["target_template"].format_map(fields))] if adapter["input_format"] == "pair" else [text]
        encoded = self.tokenizer.encode_batch(texts)
        candidates = {"input_ids": np.asarray([row.ids for row in encoded], dtype=np.int64),
                      "attention_mask": np.asarray([row.attention_mask for row in encoded], dtype=np.int64),
                      "token_type_ids": np.asarray([row.type_ids for row in encoded], dtype=np.int64)}
        logits = np.asarray(self.session.run(None, {name: candidates[name] for name in self.inputs})[0])
        if logits.ndim != 2 or not np.isfinite(logits).all():
            raise ValueError("Teacher must produce finite sequence-classification logits")
        if self.settings["backend"] == "nli":
            index = adapter["entailment_index"]
            if logits.shape[0] != 3 or logits.shape[1] <= index:
                raise ValueError("Teacher NLI output does not match the configured entailment index")
            probabilities = _softmax(logits[:, index])
        else:
            if logits.shape != (1, len(protocol["class_mapping"])):
                raise ValueError("Teacher output classes do not match class_mapping")
            probabilities = _softmax(logits[0])
        result = np.zeros(len(CHOICES), dtype=np.float64)
        for probability, name in zip(probabilities, protocol["class_mapping"]):
            result[CHOICES.index(name)] += probability
        return result.tolist()


class TeacherEngine:
    def __init__(self, settings):
        self.settings = settings
        self.runtime = None
        self.lock = threading.RLock()

    def metadata(self):
        settings = self.settings
        return {"revision": settings["revision"], "kind": "onnx_" + settings["backend"],
                "model": {"name": settings["model_name"], "revision": settings["model_revision"],
                          "sha256": settings["model"].sha256, "bytes": settings["model"].size_bytes},
                "tokenizer": {"sha256": settings["tokenizer"].sha256, "bytes": settings["tokenizer"].size_bytes},
                "protocol": settings["protocol"], "classes": list(CHOICES),
                "CPU_threads": settings["threads"], "model_loaded": self.runtime is not None}

    def label(self, request):
        matcher = MentionMatcher(request.entity, request.aliases, request.matching.model_dump())
        result = []
        for item in request.items:
            digest = input_hash(request.entity, item, request.aliases, request.matching)
            if matcher.match(item) is None:
                probabilities = [0.0] * (len(CHOICES) - 1) + [1.0]
            else:
                # A single lock covers lazy construction and inference, including
                # the tokenizer's mutable padding/truncation state.
                with self.lock:
                    if self.runtime is None:
                        self.runtime = TeacherRuntime(self.settings)
                    probabilities = self.runtime.distribution(request.entity, item)
            result.append({"comment_id": item.id, "probabilities": probabilities, "input_hash": digest})
        return {"revision": self.settings["revision"], "items": result}


SETTINGS = load_settings()
ENGINE = TeacherEngine(SETTINGS)
app = FastAPI(title="Entity Sentiment Teacher", version="1.0.0", redoc_url=None,
              swagger_ui_parameters={"validatorUrl": None})


@app.get("/health")
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
        LOGGER.exception("The configured ONNX teacher could not produce labels")
        raise HTTPException(503, "Teacher model is unavailable; check its configured assets and server logs") from error
