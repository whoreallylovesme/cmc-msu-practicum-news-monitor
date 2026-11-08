"""One joint A–F prediction from a generic llama.cpp GGUF model."""

import hashlib
import json
import logging
import math
import os
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import requests
import yaml

from lib.model_assets import asset_spec, ensure_asset, file_sha256
from lib.sentiment import CHOICES
from lib.teacher_contracts import INPUT_HASH_VERSION, input_hash


LOGGER = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
LABEL_GRAMMAR = 'root ::= "{label}"'
JOINT_GRAMMAR = 'root ::= "A" | "B" | "C" | "D" | "E" | "F"'
EVIDENCE_DELIMITER = "\nCLASS:"


def _local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_settings(path=None):
    selected = Path(path or os.environ.get("TEACHER_CONFIG", BASE / "config.yaml")).expanduser().resolve()
    config = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Teacher config must be a YAML mapping")
    source = config.get("model_weights")
    maximum = int(config.get("max_model_bytes", 16 * 1024**3))
    if isinstance(source, dict) and source.get("is_local") is True and not source.get("sha256"):
        if not isinstance(source.get("path"), str) or not source["path"].strip():
            raise ValueError("Local model_weights require a nonempty path")
        local = Path(source["path"]).expanduser()
        if not local.is_absolute():
            local = selected.parent / local
        if not local.is_file():
            raise ValueError("Local model_weights must point to an existing file")
        size = local.stat().st_size
        if not 0 < size <= maximum:
            raise ValueError("Local model_weights exceed the configured maximum size or are empty")
        source = {**source, "path": str(local.resolve()), "sha256": file_sha256(local), "size_bytes": size}
    model = asset_spec(source, max_bytes=maximum, base_dir=selected.parent)
    llama = {"binary": "llama-server", "url": None, "cpu_threads": 4, "context_size": 4096,
             "gpu_layers": 0, "parallel": 1, "startup_timeout": 180, "request_timeout": 180,
             "chat_template_kwargs": {"enable_thinking": False},
             **(config.get("llama") or {})}
    for name, minimum in (("cpu_threads", 1), ("context_size", 512), ("parallel", 1), ("gpu_layers", -1)):
        if type(llama[name]) is not int or llama[name] < minimum:
            raise ValueError(f"Invalid llama.{name}")
    if not isinstance(llama["chat_template_kwargs"], dict):
        raise ValueError("llama.chat_template_kwargs must be a mapping")
    if llama["url"] is not None and not str(llama["url"]).startswith(("http://", "https://")):
        raise ValueError("llama.url must be an HTTP(S) server URL or null")
    limits = {"comment": 2048, "parent": 512, "title": 128, **(config.get("token_limits") or {})}
    if set(limits) != {"comment", "parent", "title"} or any(type(v) is not int or v < 0 for v in limits.values()):
        raise ValueError("token_limits requires nonnegative comment, parent and title token counts")
    if limits["comment"] == 0:
        raise ValueError("The comment token limit must be positive")
    evidence = {"max_tokens": 96, "temperature": 0.0, "requested_words": 30,
                **(config.get("evidence_generation") or {})}
    if (type(evidence["max_tokens"]) is not int or evidence["max_tokens"] < 1
            or type(evidence["requested_words"]) is not int or evidence["requested_words"] < 1
            or type(evidence["temperature"]) not in (int, float)
            or not math.isfinite(evidence["temperature"]) or evidence["temperature"] < 0):
        raise ValueError("Invalid evidence generation token budget, temperature or requested word count")
    prompt = config.get("system_prompt")
    if prompt is None:
        prompt = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8")).get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("system_prompt must be a nonempty string")
    prompt = prompt.replace("{requested_words}", str(evidence["requested_words"]))
    protocol = {
        "version": "gguf-target-sentiment-v3", "input_hash_version": INPUT_HASH_VERSION,
        "system_prompt": prompt, "user_payload": ["target", "aliases", "story_title", "parent_comment", "current_comment"],
        "data_escaping": "JSON with angle brackets encoded as Unicode escapes",
        "classes": list(CHOICES), "grammar": JOINT_GRAMMAR, "label_grammar_template": LABEL_GRAMMAR,
        "truncation": {"unit": "model_tokens", "strategy": "keep_prefix", "limits": limits,
                       "prompt_limit": llama["context_size"] - 16, "hash": "full_untruncated_input"},
        "chat_template_kwargs": llama["chat_template_kwargs"],
        "llama_api": "b12192-native-completion",
        "evidence_generation": {"n_predict": evidence["max_tokens"], "temperature": evidence["temperature"],
                                "samplers": ["temperature"], "backend_sampling": False,
                                "n_probs": 0, "stream": False, "cache_prompt": True,
                                "stop": [EVIDENCE_DELIMITER], "seed": 42},
        "evidence": {"requested_words": evidence["requested_words"],
                     "completion": "require CLASS stop delimiter and nonempty EVIDENCE; no hard word-count rejection",
                     "scoring_prefix": "same_prompt_tokens + literal_evidence_and_CLASS_suffix_tokens",
                     "context_reserve": "generation_token_budget + 8 tokens before class scoring"},
        "decoding": {"n_predict": 1, "temperature": 1.0, "samplers": [], "n_probs": 64,
                     "post_sampling_probs": False, "backend_sampling": False},
        "scoring": {"initial_top_n": 64, "selected_token": "raw_logprob",
                    "missing_labels": "cached_single_label_grammar_raw_logprob", "fallback_top_n": 1,
                    "normalization": "logsumexp_across_actual_A_F_logprobs"},
        "distribution_kind": "next_label_token_likelihood_conditional_on_A_F_and_generated_evidence",
        "calibrated": False,
        "relevance": "joint learned class F; no mention-matching gate",
    }
    identity = {"protocol": protocol, "model_sha256": model.sha256,
                "model_revision": config.get("model_revision")}
    revision = "gguf-" + hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                                   separators=(",", ":")).encode()).hexdigest()
    return {"model": model, "model_name": config.get("model_name"), "model_revision": config.get("model_revision"),
            "llama": llama, "protocol": protocol, "revision": revision,
            "cache_dir": _local_path(os.environ.get("TEACHER_MODEL_CACHE_DIR") or config.get("cache_dir", ".local-data/teacher-models")),
            "log_dir": _local_path(os.environ.get("TEACHER_LOG_DIR") or config.get("log_dir", ".local-data/teacher-logs")),
            "download_timeout": float(config.get("download_timeout", 900)),
            "host": str(config.get("host", "127.0.0.1")), "port": int(config.get("port", 8082))}


def prepare_assets(settings, cache_dir=None):
    return {"model": ensure_asset(settings["model"], cache_dir or settings["cache_dir"],
                                   suffix=".gguf", timeout=settings["download_timeout"])}


class TeacherRuntime:
    def __init__(self, settings):
        self.settings = settings
        self.local = threading.local()
        self.clients = []
        self.clients_lock = threading.Lock()
        self.process = None
        self.log_handle = None
        self.url = str(settings["llama"]["url"] or "").rstrip("/")
        try:
            if not self.url:
                self._start()
            self._wait_ready()
            models = self._request("/v1/models").get("data")
            if not isinstance(models, list) or len(models) != 1 or models[0].get("id") != settings["model"].sha256:
                raise ValueError("llama.cpp must expose the configured model SHA-256 as its sole --alias")
            self.label_tokens = {label: self._tokens(label, parse_special=False) for label in CHOICES}
            if any(len(tokens) != 1 for tokens in self.label_tokens.values()):
                raise ValueError("Every class A–F must be one model token")
            self.label_tokens = {label: tokens[0] for label, tokens in self.label_tokens.items()}
            if len(set(self.label_tokens.values())) != len(CHOICES):
                raise ValueError("Class A–F tokens must be distinct")
        except Exception:
            self.close()
            raise

    def _start(self):
        model = prepare_assets(self.settings)["model"]
        options = self.settings["llama"]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        log_dir = self.settings["log_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"llama-{os.getpid()}-{time.time_ns()}.log"
        self.log_handle = log_path.open("ab")
        command = [str(options["binary"]), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
                   "--alias", self.settings["model"].sha256,
                   "--threads", str(options["cpu_threads"]),
                   "--ctx-size", str(options["context_size"] * options["parallel"]),
                   "--n-gpu-layers", str(options["gpu_layers"]), "--parallel", str(options["parallel"]),
                   "--jinja"]
        LOGGER.info("Starting teacher llama.cpp; model SHA-256 %s; log %s", self.settings["model"].sha256, log_path)
        self.process = subprocess.Popen(command, stdout=self.log_handle, stderr=subprocess.STDOUT, cwd=ROOT)

    def _client(self):
        client = getattr(self.local, "client", None)
        if client is None:
            client = requests.Session()
            with self.clients_lock:
                self.clients.append(client)
            self.local.client = client
        return client

    def _request(self, path, payload=None):
        response = self._client().request("GET" if payload is None else "POST", self.url + path,
                                        json=payload, timeout=float(self.settings["llama"]["request_timeout"]))
        response.raise_for_status()
        return response.json()

    def _wait_ready(self):
        deadline = time.monotonic() + float(self.settings["llama"]["startup_timeout"])
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("Teacher llama-server exited during startup; inspect its log")
            try:
                response = self._client().get(self.url + "/health", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.2)
        raise RuntimeError("Teacher llama-server did not become ready within startup_timeout")

    def _tokens(self, text, *, parse_special=True):
        tokens = self._request("/tokenize", {"content": text, "add_special": False,
                                             "parse_special": parse_special})["tokens"]
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError("llama.cpp returned invalid token IDs")
        return tokens

    def _truncate(self, text, limit):
        tokens = self._tokens(text, parse_special=False)
        if len(tokens) <= limit:
            return text
        return self._request("/detokenize", {"tokens": tokens[:limit]})["content"]

    def prompt(self, entity, aliases, item):
        protocol = self.settings["protocol"]
        limits = protocol["truncation"]["limits"]
        payload = {"target": entity, "aliases": list(aliases),
                   "story_title": self._truncate(item.story_title, limits["title"]),
                   "parent_comment": self._truncate(item.parent_text or "", limits["parent"]),
                   "current_comment": self._truncate(item.text, limits["comment"])}
        data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
        messages = [{"role": "system", "content": protocol["system_prompt"]},
                    {"role": "user", "content": data}]
        return self._request("/apply-template", {"messages": messages,
                             "chat_template_kwargs": protocol["chat_template_kwargs"]})["prompt"]

    def distribution(self, entity, aliases, item):
        prefix = self._tokens(self.prompt(entity, aliases, item))
        protocol = self.settings["protocol"]
        generation = protocol["evidence_generation"]
        if len(prefix) + generation["n_predict"] + 8 > protocol["truncation"]["prompt_limit"]:
            raise ValueError("The bounded prompt leaves insufficient context for complete evidence")
        result = self._request("/completion", {"prompt": prefix, **generation})
        if result.get("stop_type") != "word" or result.get("stopping_word") != EVIDENCE_DELIMITER:
            raise ValueError("Evidence did not reach the CLASS delimiter; refusing truncated reasoning")
        content = result.get("content")
        if not isinstance(content, str) or not content.strip().startswith("EVIDENCE:"):
            raise ValueError("Evidence output does not follow the requested format")
        evidence = content.strip().removeprefix("EVIDENCE:").strip()
        if not evidence:
            raise ValueError("Evidence must be nonempty; generation is bounded by the token budget")
        LOGGER.debug("Teacher evidence for %s, target %s: %s", item.id, entity, evidence)
        # Preserve the chat prefix and treat generated text as literal content:
        # echoed control-token spellings must not create conversation turns.
        suffix = self._tokens(content + EVIDENCE_DELIMITER + " ", parse_special=False)
        return self.score_prompt(prefix + suffix)

    def score_prompt(self, tokens):
        protocol = self.settings["protocol"]
        if len(tokens) > protocol["truncation"]["prompt_limit"]:
            raise ValueError("Teacher target, aliases and bounded context exceed the configured context_size")
        def completion(grammar, count):
            result = self._request("/completion", {
                "prompt": tokens, "grammar": grammar, "stream": False,
                "cache_prompt": True, **protocol["decoding"], "n_probs": count,
            })
            rows = result.get("completion_probabilities")
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                raise ValueError("llama.cpp b12192 must return one raw token probability row")
            return rows[0]

        probabilities = {}
        label_ids = set(self.label_tokens.values())

        def record(candidate):
            token = candidate.get("id")
            if token in label_ids:
                value = candidate.get("logprob")
                if type(value) not in (int, float) or not math.isfinite(value) or value > 1e-6:
                    raise ValueError("llama.cpp did not return an actual finite raw label log probability")
                probabilities[token] = float(value)

        # b12192's grammar uses rejection sampling; top probabilities after
        # sampling are not always masked. Raw top-N scores are model scores.
        row = completion(JOINT_GRAMMAR, protocol["scoring"]["initial_top_n"])
        if row.get("id") not in label_ids or not isinstance(row.get("top_logprobs"), list):
            raise ValueError("llama.cpp must return an A–F token and raw top_logprobs")
        for candidate in row["top_logprobs"]:
            record(candidate)
        record(row)
        for label in CHOICES:
            token = self.label_tokens[label]
            if token not in probabilities:
                # Raw selected-token logprob is available even outside top N.
                # Prefix caching reuses this same prompt for missing labels.
                forced = completion(LABEL_GRAMMAR.format(label=label), protocol["scoring"]["fallback_top_n"])
                if forced.get("id") != token:
                    raise ValueError(f"llama.cpp did not return forced label {label}")
                record(forced)
        log_probabilities = [probabilities[self.label_tokens[label]] for label in CHOICES]
        maximum = max(log_probabilities)
        values = [math.exp(value - maximum) for value in log_probabilities]
        total = sum(values)
        return [value / total for value in values]

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        with self.clients_lock:
            clients, self.clients = self.clients, []
        for client in clients:
            client.close()


class TeacherEngine:
    def __init__(self, settings):
        self.settings = settings
        self.runtime = None
        self.pool = None
        self.lock = threading.RLock()

    def metadata(self):
        settings = self.settings
        return {"revision": settings["revision"], "kind": "gguf_joint_target_sentiment",
                "model": {"name": settings["model_name"], "revision": settings["model_revision"],
                          "sha256": settings["model"].sha256, "bytes": settings["model"].size_bytes},
                "protocol": settings["protocol"], "classes": list(CHOICES),
                "runtime": {key: settings["llama"][key] for key in ("cpu_threads", "context_size", "gpu_layers", "parallel")},
                "external_server": bool(settings["llama"]["url"]), "model_loaded": self.runtime is not None}

    def label(self, request):
        with self.lock:
            if self.runtime is None:
                self.runtime = TeacherRuntime(self.settings)
            runtime = self.runtime

            def label_item(item):
                return {"comment_id": item.id,
                        "probabilities": runtime.distribution(request.entity, request.aliases, item),
                        "input_hash": input_hash(request.entity, item, request.aliases)}

            parallel = self.settings.get("llama", {}).get("parallel", 1)
            if parallel == 1:
                result = [label_item(item) for item in request.items]
            else:
                if self.pool is None:
                    self.pool = ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="teacher")
                futures = [self.pool.submit(label_item, item) for item in request.items]
                try:
                    result = [future.result() for future in futures]
                finally:
                    # Keep the batch lock until every worker has finished,
                    # including when one result raises an inference error.
                    for future in futures:
                        future.cancel()
                    wait(futures)
        return {"revision": self.settings["revision"], "items": result}

    def close(self):
        with self.lock:
            if self.pool is not None:
                self.pool.shutdown(wait=True)
                self.pool = None
            if self.runtime is not None:
                self.runtime.close()
                self.runtime = None
