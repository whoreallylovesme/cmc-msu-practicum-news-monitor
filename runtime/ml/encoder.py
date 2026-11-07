"""Frozen CPU encoders, including portable externally cached NLI assets."""

import hashlib
import json
import os
import re
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import requests


_RUNTIMES = OrderedDict()
_LOCK = threading.RLock()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read_weights(source, cache, suffix, maximum, timeout):
    if (
        not isinstance(source, dict) or type(source.get("is_local")) is not bool
        or not isinstance(source.get("path"), str) or not source["path"].strip()
    ):
        raise ValueError("Encoder weights require path and boolean is_local")
    expected = str(source.get("sha256", "")).lower()
    if expected and not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Encoder SHA-256 must contain 64 hexadecimal characters")
    if source["is_local"]:
        path = Path(source["path"]).expanduser()
    else:
        url = source["path"]
        if not url.startswith("https://"):
            raise ValueError("Remote encoder weights require HTTPS")
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (_digest(url.encode()) + suffix)
        if not path.is_file():
            temporary = None
            try:
                with requests.get(url, stream=True, timeout=(10, timeout)) as response:
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as handle:
                        temporary = Path(handle.name)
                        count = 0
                        for chunk in response.iter_content(1024 * 1024):
                            count += len(chunk)
                            if count > maximum:
                                raise ValueError("Encoder download exceeds configured size limit")
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                if expected and _digest(temporary.read_bytes()) != expected:
                    raise ValueError("Encoder download failed SHA-256 verification")
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"Encoder weights missing, empty or too large: {path}")
    data = path.read_bytes()
    if expected and _digest(data) != expected:
        raise ValueError(f"Encoder weights failed SHA-256 verification: {path}")
    return data


def prepare(config):
    """Download once, or read configured local files; artifacts need neither later."""
    options = config.get("student_encoder", {})
    if options.get("output_mode") == "nli":
        import nli_encoder

        return nli_encoder.prepare(config)
    cache = Path(config["model_cache_dir"]).expanduser() / "student-encoder"
    maximum = int(options.get("max_model_bytes", 128 * 1024 * 1024))
    timeout = float(options.get("download_timeout", 120))
    output_mode = options.get("output_mode", "sentence_embeddings")
    if output_mode not in ("sentence_embeddings", "sentiment_logits"):
        raise ValueError("Unknown student encoder output mode")
    padding_id = options.get("pad_token_id", 0)
    padding_token = options.get("pad_token", "[PAD]")
    if type(padding_id) is not int or padding_id < 0 or not isinstance(padding_token, str) or not padding_token:
        raise ValueError("Encoder padding requires a nonnegative token id and token text")
    weights = _read_weights(options.get("model_weights"), cache, ".onnx", maximum, timeout)
    tokenizer = _read_weights(options.get("tokenizer"), cache, ".json", 5 * 1024 * 1024, timeout)
    json.loads(tokenizer)
    return {
        "model_bytes": weights,
        "tokenizer_bytes": tokenizer,
        "model_sha256": _digest(weights),
        "tokenizer_sha256": _digest(tokenizer),
        "max_length": int(options.get("max_length", 256)),
        "batch_size": int(options.get("batch_size", 1 if output_mode == "sentiment_logits" else 16)),
        "threads": int(options.get("threads", 2)),
        "output_mode": output_mode,
        "pad_token_id": padding_id,
        "pad_token": padding_token,
        "model_source": dict(options["model_weights"]),
        "tokenizer_source": dict(options["tokenizer"]),
        "mention_options": dict(
            config.get("semantic_matching") or config.get("monitoring", {}).get("matching", {})
        ),
    }


class _Runtime:
    def __init__(self, bundle):
        import onnxruntime
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_str(bundle["tokenizer_bytes"].decode("utf-8"))
        self.tokenizer.enable_truncation(max_length=bundle["max_length"])
        self.tokenizer.enable_padding(
            pad_id=bundle.get("pad_token_id", 0), pad_token=bundle.get("pad_token", "[PAD]")
        )
        self.output_mode = bundle.get("output_mode", "sentence_embeddings")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, int(bundle["threads"]))
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            bundle["model_bytes"], sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.batch_size = max(1, int(bundle["batch_size"]))
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def encode(self, texts):
        # Repeated title/parent strings and successive training revisions reuse embeddings.
        with self.lock:
            missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                encoded = self.tokenizer.encode_batch(batch)
                candidates = {
                    "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                    "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                    "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
                }
                inputs = {name: candidates[name] for name in self.input_names}
                token_vectors = self.session.run(None, inputs)[0]
                if self.output_mode == "sentiment_logits":
                    if token_vectors.ndim != 2 or token_vectors.shape[1] != 3:
                        raise ValueError("Sentiment encoder must produce three raw logits")
                    vectors = token_vectors
                elif token_vectors.ndim == 3:
                    mask = candidates["attention_mask"][..., None].astype(np.float32)
                    vectors = (token_vectors * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
                elif token_vectors.ndim == 2:
                    vectors = token_vectors
                else:
                    raise ValueError("Encoder must produce token or sentence vectors")
                if self.output_mode != "sentiment_logits":
                    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                for text, vector in zip(batch, vectors.astype(np.float32)):
                    self.cache[text] = vector
            result = np.stack([self.cache[text] for text in texts])
            # Bound cached text and embedding memory for a long-running service.
            while len(self.cache) > 20000:
                self.cache.popitem(last=False)
            return result


def encode(bundle, texts, entity=None):
    if bundle.get("output_mode") == "nli":
        import nli_encoder

        return nli_encoder.encode(bundle, texts, entity)
    if not texts:
        dimensions = 3 if bundle.get("output_mode") == "sentiment_logits" else 384
        return np.empty((0, dimensions), dtype=np.float32)
    key = (
        bundle["model_sha256"], bundle["tokenizer_sha256"], bundle["max_length"],
        bundle["threads"], bundle["batch_size"], bundle.get("output_mode", "sentence_embeddings"),
        bundle.get("pad_token_id", 0), bundle.get("pad_token", "[PAD]"),
    )
    with _LOCK:
        if key not in _RUNTIMES:
            _RUNTIMES[key] = _Runtime(bundle)
        runtime = _RUNTIMES[key]
        _RUNTIMES.move_to_end(key)
        while len(_RUNTIMES) > 2:
            _RUNTIMES.popitem(last=False)
    return runtime.encode(texts)


def features(bundle, items, entity, aliases=()):
    if bundle.get("output_mode") == "nli":
        import nli_encoder

        return nli_encoder.features(bundle, items, entity, aliases)
    if bundle.get("output_mode") == "sentiment_logits":
        from lib.mentions import MentionMatcher

        if not items:
            empty = np.empty((0, 8), dtype=np.float32)
            return empty, empty.copy()
        comments = [str(item.get("text") or "") for item in items]
        # The pretrained model card specifies URL and username placeholders.
        processed = [" ".join(
            "@user" if word.startswith("@") and len(word) > 1
            else "http" if word.startswith("http") else word
            for word in text.split(" ")
        ) for text in comments]
        logits = encode(bundle, processed)
        matcher = MentionMatcher(entity, aliases, bundle.get("mention_options", {}))
        kinds = [matcher.match(item) for item in items]
        extra = np.asarray([
            [min(len(text.split()), 150) / 150]
            + [float(kind == name) for name in ("direct", "alias", "typo", "context")]
            for text, kind in zip(comments, kinds)
        ], dtype=np.float32)
        # Parent/title content can resolve the scope flag, never the tone.
        matrix = np.column_stack([logits, extra])
        return matrix, matrix.copy()
    comments = [str(item.get("text") or "")[:2400] for item in items]
    parents = [str(item.get("parent_text") or "")[:1200] for item in items]
    titles = [str(item.get("story_title") or "")[:320] for item in items]
    combined = encode(bundle, comments + parents + titles)
    comment_vectors, parent_vectors, title_vectors = np.split(combined, 3)
    # Missing context is zero, rather than the embedding of an empty sentence.
    parent_vectors *= np.asarray([bool(text.strip()) for text in parents])[:, None]
    title_vectors *= np.asarray([bool(text.strip()) for text in titles])[:, None]
    patterns = [re.compile(r"(?<!\w)" + re.escape(name).replace(r"\ ", r"\s+") + r"(?!\w)", re.I)
                for name in [entity, *aliases] if name]
    mentions = np.asarray([
        [float(any(pattern.search(text) for pattern in patterns)) for text in fields]
        + [min(len(fields[0].split()), 150) / 150, float(bool(fields[1].strip()))]
        for fields in zip(comments, parents, titles)
    ], dtype=np.float32)
    # Context resolves pronouns; separate low-weight fields prevent topic/title dominance.
    relevance = np.column_stack([comment_vectors, 0.5 * parent_vectors, 0.25 * title_vectors, mentions])
    sentiment = np.column_stack([comment_vectors, 0.2 * parent_vectors, 0.05 * title_vectors, mentions])
    return relevance, sentiment
