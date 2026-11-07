"""Portable, frozen NLI features for sentiment about the configured entity.

Artifacts contain pinned asset descriptors and a small calibration head. Model
files live in an external checksum-addressed cache; ONNX Runtime reads a path.
"""

import hashlib
import json
import os
import re
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import requests


BASE = Path(__file__).resolve().parent
HYPOTHESIS_TEMPLATES = (
    "The author expresses a negative opinion about {entity}.",
    "The author makes a neutral or factual statement about {entity}.",
    "The author expresses a positive opinion about {entity}.",
)
_CACHE_LIMIT = 20000
_LOCK = threading.RLock()
_VERIFIED = OrderedDict()
_PREPARED_CACHES = OrderedDict()
# Absolute deployment paths belong only to this process, never to an artifact.
_CACHE_OVERRIDES = {}
_SOURCE_OVERRIDES = {}


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum(value):
    result = str(value or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise ValueError("NLI assets require a pinned 64-character SHA-256")
    return result


def _identity(bundle):
    return bundle["model_sha256"], bundle["tokenizer_sha256"]


def _cache_dir(bundle, selected=None):
    path = selected or os.environ.get("ML_ENCODER_CACHE_DIR") or _CACHE_OVERRIDES.get(_identity(bundle))
    if path:
        return Path(path).expanduser()
    data = Path(os.environ.get("MONITOR_DATA_DIR", BASE.parents[1] / ".local-data/ml"))
    return data / "encoder-weights"


def _source(options, name):
    source = options.get("model_weights" if name == "model" else "tokenizer")
    if (not isinstance(source, dict) or type(source.get("is_local")) is not bool
            or not isinstance(source.get("path"), str) or not source["path"].strip()):
        raise ValueError("NLI assets require path, boolean is_local and sha256")
    digest = _checksum(source.get("sha256"))
    path = source["path"]
    if not source["is_local"] and not path.startswith("https://"):
        raise ValueError("Remote NLI assets require HTTPS")
    local = Path(path).expanduser() if source["is_local"] else None
    if local is not None and not local.is_absolute():
        local = BASE / local
    size = options.get(name + "_size_bytes", source.get("size_bytes", source.get("bytes")))
    if size is None:
        size = local.stat().st_size if local is not None else None
    if type(size) is not int or size <= 0:
        raise ValueError(f"Set student_encoder.{name}_size_bytes for this pinned NLI asset")
    # Outside local paths are useful during preparation, but cannot travel to
    # another host. Their verified copies use the same portable cache filenames.
    portable = path
    if local is not None:
        try:
            portable = str(local.resolve().relative_to(BASE))
        except ValueError:
            portable = local.name
    return {"path": portable, "is_local": source["is_local"], "sha256": digest,
            "size_bytes": size}, local


def prepare(config):
    """Build a small descriptor; loading a cached feature needs no NN session."""
    options = config.get("student_encoder", {})
    model, local_model = _source(options, "model")
    tokenizer, local_tokenizer = _source(options, "tokenizer")
    maximum = int(options.get("max_model_bytes", 1024 * 1024 * 1024))
    if model["size_bytes"] > maximum or tokenizer["size_bytes"] > 32 * 1024 * 1024:
        raise ValueError("NLI asset exceeds its configured size limit")
    templates = options.get("hypothesis_templates", options.get("hypotheses", HYPOTHESIS_TEMPLATES))
    if (not isinstance(templates, (list, tuple)) or len(templates) != 3
            or any(not isinstance(s, str) or "{entity}" not in s for s in templates)):
        raise ValueError("NLI requires three hypothesis templates containing {entity}")
    length, threads = options.get("max_length", 512), options.get("threads", 2)
    entailment = options.get("entailment_index", 0)
    if (type(length) is not int or not 8 <= length <= 512
            or type(threads) is not int or threads < 1
            or type(entailment) is not int or entailment < 0):
        raise ValueError("Invalid NLI token limit, CPU thread count or entailment index")
    feature_cache = options.get("feature_cache")
    if feature_cache:
        feature_cache = str(feature_cache)
        if not Path(feature_cache).is_absolute() and ".." in Path(feature_cache).parts:
            raise ValueError("Use an absolute path for an NLI feature cache outside runtime/ml")
    bundle = {
        "output_mode": "nli", "descriptor_version": 1,
        "model_source": model, "tokenizer_source": tokenizer,
        "model_sha256": model["sha256"], "tokenizer_sha256": tokenizer["sha256"],
        "model_size_bytes": model["size_bytes"], "tokenizer_size_bytes": tokenizer["size_bytes"],
        "max_length": length, "threads": threads, "batch_size": 1,
        "pad_token_id": 0, "pad_token": "[PAD]", "entailment_index": entailment,
        "hypothesis_templates": list(templates),
        "download_timeout": float(options.get("download_timeout", 120)),
        "feature_cache": feature_cache or None,
        "feature_cache_sha256": (_checksum(options["feature_cache_sha256"])
                                 if options.get("feature_cache_sha256") else None),
        "entity": str(config.get("semantic_entity") or config.get("monitoring", {}).get("query") or ""),
        "mention_options": dict(config.get("semantic_matching")
                                or config.get("monitoring", {}).get("matching", {})),
    }
    identity = _identity(bundle)
    selected = options.get("cache_dir") or config.get("encoder_cache_dir")
    with _LOCK:
        if selected:
            _CACHE_OVERRIDES[identity] = str(Path(selected).expanduser())
        _SOURCE_OVERRIDES[identity] = {"model": local_model, "tokenizer": local_tokenizer}
    return bundle


@contextmanager
def _asset_lock(path):
    # Atomic rename protects readers; a file lock also avoids duplicate downloads
    # when the preflight command and service workers start concurrently.
    with path.open("a+b") as handle:
        try:
            import fcntl
        except ImportError:
            yield
        else:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _verified(path, source):
    if not path.is_file():
        return False
    stat = path.stat()
    if stat.st_size != source["size_bytes"]:
        return False
    key = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns,
           stat.st_size, source["sha256"])
    if key not in _VERIFIED:
        if _file_sha(path) != source["sha256"]:
            return False
        _VERIFIED[key] = True
        while len(_VERIFIED) > 16:
            _VERIFIED.popitem(last=False)
    _VERIFIED.move_to_end(key)
    return True


def _ensure_asset(bundle, name, cache):
    source = bundle[name + "_source"]
    suffix = ".onnx" if name == "model" else ".json"
    target = cache / (source["sha256"] + suffix)
    with _asset_lock(cache / (source["sha256"] + ".lock")):
        if _verified(target, source):
            return target
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                if source["is_local"]:
                    local = _SOURCE_OVERRIDES.get(_identity(bundle), {}).get(name)
                    local = local or BASE / source["path"]
                    with Path(local).open("rb") as original:
                        count = 0
                        for chunk in iter(lambda: original.read(1024 * 1024), b""):
                            count += len(chunk)
                            if count > source["size_bytes"]:
                                raise ValueError("Local NLI asset exceeds its pinned byte size")
                            handle.write(chunk)
                else:
                    with requests.get(source["path"], stream=True,
                                      timeout=(10, bundle.get("download_timeout", 120))) as response:
                        response.raise_for_status()
                        count = 0
                        for chunk in response.iter_content(1024 * 1024):
                            count += len(chunk)
                            if count > source["size_bytes"]:
                                raise ValueError("NLI download exceeds its pinned byte size")
                            handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if not _verified(temporary, source):
                raise ValueError(f"NLI {name} failed SHA-256 or byte-size verification")
            os.replace(temporary, target)
            if not _verified(target, source):
                raise ValueError(f"NLI {name} failed verification after installation")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return target


def ensure_assets(config_or_bundle, cache_dir=None):
    """Verify/install pinned files and return paths for preflight or Docker builds."""
    bundle = (config_or_bundle if config_or_bundle.get("output_mode") == "nli"
              else prepare(config_or_bundle))
    cache = _cache_dir(bundle, cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        return {name: _ensure_asset(bundle, name, cache) for name in ("model", "tokenizer")}


def hypotheses(bundle, entity):
    if not isinstance(entity, str) or not entity.strip():
        raise ValueError("NLI features require the configured entity")
    return [template.replace("{entity}", entity) for template in bundle["hypothesis_templates"]]


def feature_key(bundle, entity, text):
    """Cache identity includes actual text and target, never IDs or labels."""
    return _sha(_json_bytes([bundle["model_sha256"], bundle["tokenizer_sha256"],
                             hypotheses(bundle, entity), bundle["max_length"], entity, text,
                             bundle.get("entailment_index", 0)]))


def _metadata(bundle):
    return {"schema_version": 1, "output_mode": "nli", "model_sha256": bundle["model_sha256"],
            "tokenizer_sha256": bundle["tokenizer_sha256"],
            "hypothesis_templates": bundle["hypothesis_templates"],
            "max_length": bundle["max_length"], "entailment_index": bundle.get("entailment_index", 0),
            "padding": {"id": 0, "token": "[PAD]"}, "premise": "current_comment_only"}


def _vector(value):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        raise ValueError("Prepared NLI features must contain exactly three finite logits")
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Prepared NLI features must contain exactly three finite logits")
    return result


def _prepared_features(bundle):
    selected = bundle.get("feature_cache")
    if not selected:
        return {}
    path = BASE / selected
    try:
        stat = path.stat()
        if stat.st_size > 32 * 1024 * 1024:
            return {}
        signature = (str(path.resolve()), stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size,
                     _sha(_json_bytes(_metadata(bundle))), bundle.get("feature_cache_sha256"))
        with _LOCK:
            if signature in _PREPARED_CACHES:
                _PREPARED_CACHES.move_to_end(signature)
                return _PREPARED_CACHES[signature]
            raw = path.read_bytes()
            if bundle.get("feature_cache_sha256") and _sha(raw) != bundle["feature_cache_sha256"]:
                return {}
            document = json.loads(raw)
            meta, rows = document["metadata"], document["features"]
            if (not isinstance(rows, dict) or len(rows) > _CACHE_LIMIT
                    or document["metadata_sha256"] != _sha(_json_bytes(meta))
                    or meta != dict(_metadata(bundle), features_sha256=_sha(_json_bytes(rows)))):
                return {}
            vectors = {_checksum(key): _vector(value) for key, value in rows.items()}
            _PREPARED_CACHES[signature] = vectors
            while len(_PREPARED_CACHES) > 2:
                _PREPARED_CACHES.popitem(last=False)
            return vectors
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        # A stale/missing prepared file is an optimization miss, never a source
        # of predictions for a different model, target or changed comment.
        return {}


class _Runtime:
    def __init__(self, bundle):
        import onnxruntime
        from tokenizers import Tokenizer

        assets = ensure_assets(bundle)
        self.tokenizer = Tokenizer.from_file(str(assets["tokenizer"]))
        self.tokenizer.enable_truncation(max_length=bundle["max_length"], strategy="only_first")
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = bundle["threads"]
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(assets["model"]), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = {value.name for value in self.session.get_inputs()}
        self.bundle = bundle
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.inference_calls = 0

    def encode(self, texts, entity):
        result = []
        with self.lock:
            hypothesis = hypotheses(self.bundle, entity)
            for text in texts:
                key = feature_key(self.bundle, entity, text)
                if key not in self.cache:
                    encoded = self.tokenizer.encode_batch([(text, h) for h in hypothesis])
                    candidates = {
                        "input_ids": np.asarray([row.ids for row in encoded], dtype=np.int64),
                        "attention_mask": np.asarray([row.attention_mask for row in encoded], dtype=np.int64),
                        "token_type_ids": np.asarray([row.type_ids for row in encoded], dtype=np.int64),
                    }
                    logits = self.session.run(None, {name: candidates[name] for name in self.input_names})[0]
                    index = self.bundle.get("entailment_index", 0)
                    if (logits.ndim != 2 or logits.shape[0] != 3 or logits.shape[1] <= index
                            or not np.isfinite(logits).all()):
                        raise ValueError("NLI encoder must produce finite entailment logits for three hypotheses")
                    self.cache[key] = logits[:, index].astype(np.float32)
                    self.inference_calls += 1
                    while len(self.cache) > _CACHE_LIMIT:
                        self.cache.popitem(last=False)
                self.cache.move_to_end(key)
                result.append(self.cache[key])
        return np.stack(result) if result else np.empty((0, 3), dtype=np.float32)


def _runtime(bundle):
    # Share the legacy encoder's two-session LRU so switching encoder modes
    # cannot accidentally retain another two large neural models.
    import encoder

    key = ("nli", bundle["model_sha256"], bundle["tokenizer_sha256"], bundle["max_length"],
           bundle["threads"], tuple(bundle["hypothesis_templates"]), bundle.get("entailment_index", 0))
    with encoder._LOCK:
        if key not in encoder._RUNTIMES:
            encoder._RUNTIMES[key] = _Runtime(bundle)
        runtime = encoder._RUNTIMES[key]
        encoder._RUNTIMES.move_to_end(key)
        while len(encoder._RUNTIMES) > 2:
            encoder._RUNTIMES.popitem(last=False)
    return runtime


def encode(bundle, texts, entity=None, *, use_prepared_cache=True):
    texts = [str(text or "") for text in texts]
    if not texts:
        return np.empty((0, 3), dtype=np.float32)
    entity = entity or bundle.get("entity")
    prepared = _prepared_features(bundle) if use_prepared_cache else {}
    keys = [feature_key(bundle, entity, text) for text in texts]
    found = {key: prepared[key] for key in keys if key in prepared}
    missing = dict((key, text) for key, text in zip(keys, texts) if key not in found)
    if missing:
        values = _runtime(bundle).encode(list(missing.values()), entity)
        found.update(zip(missing, values))
    return np.stack([found[key] for key in keys]).astype(np.float32)


def features(bundle, items, entity, aliases=()):
    from lib.mentions import MentionMatcher

    if not items:
        empty = np.empty((0, 8), dtype=np.float32)
        return empty, empty.copy()
    comments = [str(item.get("text") or "") for item in items]
    logits = encode(bundle, comments, entity)
    matcher = MentionMatcher(entity, aliases, bundle.get("mention_options", {}))
    kinds = [matcher.match(item) for item in items]
    extra = np.asarray([
        [min(len(text.split()), 150) / 150]
        + [float(kind == name) for name in ("direct", "alias", "typo", "context")]
        for text, kind in zip(comments, kinds)
    ], dtype=np.float32)
    matrix = np.column_stack([logits, extra])
    return matrix, matrix.copy()


def save_feature_cache(bundle, vectors, destination):
    """Atomically save only protocol metadata, hashed text keys and three logits."""
    if len(vectors) > _CACHE_LIMIT:
        raise ValueError("Prepared NLI feature cache exceeds 20000 rows")
    rows = {_checksum(key): _vector(value).tolist() for key, value in vectors.items()}
    meta = dict(_metadata(bundle), features_sha256=_sha(_json_bytes(rows)))
    document = {"metadata": meta, "metadata_sha256": _sha(_json_bytes(meta)), "features": rows}
    payload = _json_bytes(document) + b"\n"
    path = Path(destination).expanduser()
    if not path.is_absolute():
        path = BASE / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "rows": len(rows), "sha256": _sha(payload), "bytes": len(payload)}


def generate_feature_cache(config_or_bundle, items, destination=None, entity=None, *, precomputed=None):
    """Build a cache from raw comments, optionally reusing verified earlier logits.

    ``precomputed`` maps ``feature_key(bundle, entity, text)`` to three logits.
    No comment IDs, reference labels, teacher targets or calibration predictions
    are read or written by this helper.
    """
    bundle = (config_or_bundle if config_or_bundle.get("output_mode") == "nli"
              else prepare(config_or_bundle))
    entity = entity or bundle.get("entity")
    required = {feature_key(bundle, entity, str(item.get("text") or "")):
                str(item.get("text") or "") for item in items}
    if len(required) > _CACHE_LIMIT:
        raise ValueError("Prepared NLI feature cache exceeds 20000 rows")
    precomputed = precomputed or {}
    vectors = {key: _vector(precomputed[key]) for key in required if key in precomputed}
    missing = {key: text for key, text in required.items() if key not in vectors}
    if missing:
        values = encode(bundle, list(missing.values()), entity, use_prepared_cache=False)
        vectors.update(zip(missing, values))
    selected = destination or bundle.get("feature_cache")
    if not selected:
        raise ValueError("Choose a destination for the prepared NLI feature cache")
    report = save_feature_cache(bundle, vectors, selected)
    return dict(report, reused_rows=len(required) - len(missing), computed_rows=len(missing))
