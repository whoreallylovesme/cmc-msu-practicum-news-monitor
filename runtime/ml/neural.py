"""ModernBERT training with compact adapters and pinned encoder-weight overrides."""

import copy
from contextlib import nullcontext
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import urljoin, urlsplit

import numpy as np

from lib.model_assets import asset_spec, ensure_asset, file_sha256, verify_asset
from lib.sentiment import CHOICES, OFF_TOPIC


LOGGER = logging.getLogger(__name__)
MODEL_LOCK = threading.RLock()
INPUT_FORMAT = "target-comment-parent-title-v1"
COMPANIONS = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
_RUNTIME = None
ROOT = Path(__file__).resolve().parents[2]
ENCODER_PREFIX = "base_model.model.model.layers."


def prepare(config):
    settings = copy.deepcopy(config["student_encoder"])
    weights = settings["model_weights"]
    if weights.get("is_local") and Path(weights["path"]).expanduser().is_dir():
        weights["path"] = str(Path(weights["path"]).expanduser() / "model.safetensors")
    if not weights.get("is_local"):
        parts = urlsplit(weights["path"]).path.split("/")
        if "resolve" in parts:
            revision = parts[parts.index("resolve") + 1]
            if len(revision) != 40 or any(value not in "0123456789abcdef" for value in revision.lower()):
                raise ValueError("Hugging Face assets require an immutable commit revision")
    files = {"model.safetensors": weights}
    for name in COMPANIONS:
        spec = copy.deepcopy(settings.get("companion_assets", {}).get(name, {}))
        spec.setdefault("is_local", weights["is_local"])
        if "path" not in spec:
            spec["path"] = (str(Path(weights["path"]).expanduser().parent / name)
                            if spec["is_local"] else urljoin(weights["path"], name))
        files[name] = spec
    for name, spec in files.items():
        if spec["is_local"]:
            path = Path(spec["path"]).expanduser().resolve()
            spec["path"] = str(path)
            if not spec.get("sha256"):
                spec.update(sha256=file_sha256(path), size_bytes=path.stat().st_size)
        maximum = int(settings.get("max_model_bytes", 2 * 1024 ** 3)) if name == "model.safetensors" else 16 * 1024 ** 2
        checked = asset_spec(spec, max_bytes=maximum)
        files[name] = {
            "path": checked.path, "is_local": checked.is_local,
            "sha256": checked.sha256, "size_bytes": checked.size_bytes,
        }
    return {
        "files": files,
        "cache_dir": str(Path(settings.get("cache_dir", ".local-data/student-models")).expanduser()),
        "data_dir": str(config.get("data_dir") or ".local-data/ml"),
        "download_timeout": float(settings.get("download_timeout", 180)),
        "max_model_bytes": int(settings.get("max_model_bytes", 2 * 1024 ** 3)),
        "device": settings.get("device", "auto"),
        "dtype": settings.get("dtype", "float32"),
        "threads": int(settings.get("threads", 4)),
        "max_length": int(settings.get("max_length", 512)),
        "batch_size": int(settings.get("batch_size", 4)),
    }


def _base_identity(bundle):
    return {name: spec["sha256"] for name, spec in bundle["files"].items()}


def _model_directory(bundle):
    digest = hashlib.sha256(json.dumps(_base_identity(bundle), sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get("ML_ENCODER_CACHE_DIR", bundle["cache_dir"]))
    directory = cache / digest
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in bundle["files"].items():
        spec = asset_spec(source, max_bytes=bundle["max_model_bytes"] if name == "model.safetensors" else 16 * 1024 ** 2)
        path = ensure_asset(spec, cache / "assets", suffix=Path(name).suffix, timeout=bundle["download_timeout"])
        destination = directory / name
        if not destination.is_symlink() or destination.resolve() != path.resolve():
            temporary = directory / (name + ".link")
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(path.resolve())
            os.replace(temporary, destination)
    with (directory / "config.json").open() as handle:
        if json.load(handle).get("model_type") != "modernbert":
            raise ValueError("The student requires a ModernBERT encoder checkpoint")
    return directory


def _device(bundle, torch):
    value = bundle["device"]
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if value not in {"cpu", "mps", "cuda"}:
        raise ValueError("Student device must be auto, cpu, mps or cuda")
    return value


def _encoder_parameters(model):
    return {name: parameter for name, parameter in model.named_parameters()
            if name.startswith(ENCODER_PREFIX) and ".lora_" not in name
            and ".original_module." not in name}


def _encoder_layer(name):
    return int(name[len(ENCODER_PREFIX):].split(".", 1)[0])


def _encoder_names(model, layers):
    count = len(model.base_model.model.model.layers)
    if (not isinstance(layers, list) or any(type(index) is not int or not 0 <= index < count for index in layers)
            or layers != sorted(set(layers))):
        raise ValueError("Encoder snapshot has an invalid block scope")
    return {name: parameter for name, parameter in _encoder_parameters(model).items()
            if _encoder_layer(name) in layers}


def _encoder_data_dir(bundle):
    directory = Path(os.environ.get("MONITOR_DATA_DIR", bundle.get("data_dir", ".local-data/ml"))).expanduser()
    return directory if directory.is_absolute() else ROOT / directory


def _restore_encoder(model, bundle, encoder_state):
    import torch
    from safetensors import safe_open

    expected = _encoder_names(model, encoder_state["layers"])
    keys = encoder_state["keys"]
    if not expected or not isinstance(keys, list) or keys != sorted(expected):
        raise ValueError("Encoder snapshot must contain exactly the parameters of its recorded blocks")
    spec = asset_spec(encoder_state["asset"], max_bytes=bundle["max_model_bytes"], base_dir=_encoder_data_dir(bundle))
    if spec.is_local:
        path = Path(spec.path)
        if not verify_asset(path, spec):
            raise ValueError("Encoder snapshot failed SHA-256 or byte-size verification")
    else:
        path = ensure_asset(spec, _encoder_data_dir(bundle) / "encoder-checkpoints",
                            suffix=".safetensors", timeout=bundle["download_timeout"])
    with safe_open(path, framework="pt", device="cpu") as tensors, torch.no_grad():
        if set(tensors.keys()) != set(expected):
            raise ValueError("Encoder snapshot tensor keys do not match its recorded scope")
        for name, parameter in expected.items():
            tensor = tensors.get_tensor(name)
            if tensor.dtype != torch.float32 or tensor.shape != parameter.shape:
                raise ValueError("Encoder snapshot requires exact float32 parameter shapes")
            parameter.copy_(tensor)


def _save_encoder(model, bundle, keys):
    from safetensors.torch import save_file

    parameters = _encoder_parameters(model)
    tensors = {name: parameters[name].detach().float().cpu().contiguous() for name in keys}
    directory = _encoder_data_dir(bundle) / "encoder-checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_file(tensors, str(temporary), metadata={"format": "pt"})
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        checksum, count = file_sha256(temporary), temporary.stat().st_size
        destination = directory / (checksum + ".safetensors")
        temporary.chmod(0o444)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"asset": {"path": str(Path("encoder-checkpoints") / destination.name), "is_local": True,
                      "sha256": checksum, "size_bytes": count},
            "keys": list(keys), "layers": sorted({_encoder_layer(name) for name in keys})}


def _dropout_overrides(value):
    if value is None:
        return {}
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("Student dropout must be null or a finite number in [0, 1)")
    return {name: float(value) for name in (
        "embedding_dropout", "attention_dropout", "mlp_dropout", "classifier_dropout",
    )}


def _probability_model(value):
    if not isinstance(value, str) or value not in {"flat", "coarse-factorized"}:
        raise ValueError("Student probability_model must be flat or coarse-factorized")
    return value


def _load(bundle, lora, state=None, encoder_state=None, unfreeze_last_n_layers=0, dropout=None,
          probability_model="flat"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    torch.set_num_threads(bundle["threads"])
    directory = _model_directory(bundle)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        directory, num_labels=4 if _probability_model(probability_model) == "coarse-factorized" else len(CHOICES),
        local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="sdpa", reference_compile=False,
        ignore_mismatched_sizes=True, output_loading_info=True,
        **_dropout_overrides(dropout),
    )
    classifier = {"classifier.weight", "classifier.bias"}
    mismatched = set(loading["mismatched_keys"])
    missing = set(loading["missing_keys"])
    unexpected = set(loading["unexpected_keys"])
    if (mismatched - classifier or missing - classifier
            or unexpected - {"decoder.weight", "decoder.bias"} or loading.get("error_msgs")):
        raise ValueError("Student checkpoint may only initialize or resize its output classifier; encoder and head must match")
    if mismatched:
        LOGGER.info("Initialized %d-class output classifier from a checkpoint with a different label count", model.config.num_labels)
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and name.endswith((".attn.Wqkv", ".attn.Wo"))]
    if not targets:
        raise ValueError("ModernBERT attention projections were not found")
    heads = [name for name in ("head", "classifier") if hasattr(model, name)]
    model = get_peft_model(model, LoraConfig(
        task_type="SEQ_CLS", r=lora["rank"], lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"], target_modules=targets, modules_to_save=heads, bias="none",
    ))
    if state is not None:
        if set(state) != set(get_peft_model_state_dict(model)):
            raise ValueError("Student adapter snapshot does not match its architecture")
        set_peft_model_state_dict(model, {name: torch.from_numpy(value.copy()) for name, value in state.items()})
    count = len(model.base_model.model.model.layers)
    if type(unfreeze_last_n_layers) is not int or not 0 <= unfreeze_last_n_layers <= count:
        raise ValueError(f"unfreeze_last_n_layers must be an integer between zero and {count}")
    if (encoder_state is not None or unfreeze_last_n_layers) and bundle["dtype"] != "float32":
        raise ValueError("Encoder updates require float32 model parameters; use amp_dtype for mixed precision")
    if encoder_state is not None:
        _restore_encoder(model, bundle, encoder_state)
    for parameter in _encoder_names(model, list(range(count - unfreeze_last_n_layers, count))).values():
        parameter.requires_grad_(True)
    device = _device(bundle, torch)
    if bundle["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("Student dtype must be float32 or bfloat16")
    model.to(device=device, dtype=getattr(torch, bundle["dtype"]))
    return model, tokenizer, device


def _examples(items, entity, aliases):
    targets, documents = [], []
    for item in items:
        target = str(item.get("entity") or entity).strip()
        if not target:
            raise ValueError("The student needs a target entity")
        known_as = aliases if target.casefold() == entity.casefold() else []
        targets.append("Target entity: " + target + ("\nAliases: " + ", ".join(known_as) if known_as else ""))
        documents.append(
            "Comment:\n" + str(item.get("text") or "")
            + "\n\nParent comment (context only):\n" + str(item.get("parent_text") or "")
            + "\n\nStory title (context only):\n" + str(item.get("story_title") or "")
        )
    return targets, documents


def _batch(tokenizer, items, entity, aliases, bundle, device):
    targets, documents = _examples(items, entity, aliases)
    encoded = tokenizer(
        targets, documents, max_length=bundle["max_length"], truncation="only_second",
        padding=True, return_tensors="pt", return_token_type_ids=False,
    )
    return {name: value.to(device) for name, value in encoded.items()}


def _state(model):
    from peft import get_peft_model_state_dict

    return {name: value.detach().float().cpu().numpy().copy()
            for name, value in get_peft_model_state_dict(model).items()}


def _identity(artifact):
    encoder_state = artifact.get("encoder_state")
    digest = hashlib.sha256(json.dumps({
        "base": _base_identity(artifact["encoder"]), "lora": artifact["lora"],
        "encoder_state": None if encoder_state is None else {
            "sha256": encoder_state["asset"]["sha256"], "size_bytes": encoder_state["asset"]["size_bytes"],
            "keys": encoder_state["keys"], "layers": encoder_state["layers"],
        },
        "entity": artifact["entity"], "aliases": artifact["aliases"],
        "input_format": artifact["input_format"], "training": artifact.get("training"), "runtime": {
            name: artifact["encoder"][name] for name in ("device", "dtype", "max_length", "threads")
        },
    }, sort_keys=True).encode())
    probability_model = _probability_model(artifact.get("probability_model", "flat"))
    if probability_model != "flat":
        digest.update(b"probability_model=" + probability_model.encode())
    for name, value in sorted(artifact["state"].items()):
        digest.update(name.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _log_probabilities(logits, torch, probability_model="flat", temperature=1.0):
    if _probability_model(probability_model) == "flat":
        return torch.log_softmax(logits.float() / temperature, dim=-1)
    if logits.shape[-1] != 4:
        raise ValueError("Coarse-factorized student requires four output logits")
    logits = logits.float()
    joint = torch.cat((torch.nn.functional.logsigmoid(-logits[:, 3:])
                       + torch.log_softmax(logits[:, :3], dim=-1),
                       torch.nn.functional.logsigmoid(logits[:, 3:])), dim=-1)
    return joint if temperature == 1 else torch.log_softmax(joint / temperature, dim=-1)


def _loss(logits, expected, partial, temperature, torch, sample_weight=None, balance_weight=None,
          probability_model="flat", relevance_only=None):
    log_probabilities = _log_probabilities(logits, torch, probability_model, temperature)
    if temperature != 1.0:
        expected = expected.pow(1.0 / temperature)
        expected = expected / expected.sum(dim=-1, keepdim=True)
    losses = torch.nn.functional.kl_div(log_probabilities, expected, reduction="none").sum(dim=-1)
    if probability_model == "flat" and partial.any():
        grouped = torch.stack([
            torch.logsumexp(log_probabilities[:, :2], dim=-1), log_probabilities[:, 2],
            torch.logsumexp(log_probabilities[:, 3:OFF_TOPIC], dim=-1), log_probabilities[:, OFF_TOPIC],
        ], dim=-1)
        reference = torch.stack([expected[:, :2].sum(dim=-1), expected[:, 2],
                                 expected[:, 3:OFF_TOPIC].sum(dim=-1), expected[:, OFF_TOPIC]], dim=-1)
        marginal = torch.nn.functional.kl_div(grouped, reference, reduction="none").sum(dim=-1)
        losses = torch.where(partial, marginal, losses)
    if relevance_only is not None and relevance_only.any():
        log_relevant = torch.logsumexp(log_probabilities[:, :OFF_TOPIC], dim=-1)
        log_off_topic = log_probabilities[:, OFF_TOPIC]
        expected_off_topic = expected[:, OFF_TOPIC]
        binary = -(expected_off_topic * log_off_topic + (1 - expected_off_topic) * log_relevant)
        losses = torch.where(relevance_only, binary, losses)
    if sample_weight is not None:
        # Confidence scales each example's contribution; it is not cancelled
        # by renormalizing a low-confidence minibatch to unit total weight.
        losses = losses * sample_weight
    if balance_weight is not None:
        losses = losses * balance_weight
    return losses.mean() * temperature ** 2


def _training_targets(items, targets, probability_model="flat"):
    probability_model = _probability_model(probability_model)
    targets = np.asarray(targets, dtype=np.float32)
    if targets.shape != (len(items), len(CHOICES)) or not len(items) or not np.isfinite(targets).all() or (targets < 0).any() or (targets.sum(axis=1) <= 0).any():
        raise ValueError("Training needs finite nonnegative six-class targets")
    targets = targets / targets.sum(axis=1, keepdims=True)
    # Older NewsMTSC inputs use polarity_only; their F mass is zero and their
    # three-polarity objective is unchanged by the extra zero-mass group.
    partial = np.asarray([bool(item.get("partial_labels", item.get("polarity_only", False))) for item in items])
    relevance_only = np.asarray([bool(item.get("relevance_only", False)) for item in items])
    if (partial & relevance_only).any():
        raise ValueError("relevance_only cannot be combined with partial polarity labels")
    if relevance_only.any() and probability_model != "flat":
        raise ValueError("relevance_only currently requires the flat probability model")
    if probability_model == "coarse-factorized":
        targets = np.column_stack((targets[:, :2].sum(axis=1), targets[:, 2],
                                   targets[:, 3:OFF_TOPIC].sum(axis=1), targets[:, OFF_TOPIC]))
        # All four coarse classes are observed after projection; intensity is
        # neither predicted nor required by this objective.
        partial = np.zeros(len(items), dtype=bool)
    weights = np.asarray([item.get("sample_weight", 1.0) for item in items], dtype=np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any() or (weights > 1).any():
        raise ValueError("sample_weight must be finite, positive and at most one")
    weights = weights.astype(np.float32)
    if (weights <= 0).any():
        raise ValueError("sample_weight must remain positive at float32 precision")
    return targets, partial, weights, relevance_only


def _weight_summary(values):
    values = np.asarray(values, dtype="<f4")
    return {"sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "format": "little-endian-float32-in-training-item-order-v1", "items": len(values),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "sum": float(values.astype(np.float64).sum())}


def _balance_weights(items, targets, partial, settings):
    """Whole-example TRAIN weights; keep soft targets and confidence unchanged."""
    spec = settings.get("training_balance")
    if spec is None:
        return np.ones(len(items), dtype=np.float32), None
    if not isinstance(spec, dict):
        raise ValueError("training_balance must be null or a mapping")
    exponent, maximum = spec.get("exponent", .5), spec.get("max_class_weight", 4.0)
    source_weights = spec.get("source_weights", {})
    if (isinstance(exponent, bool) or not isinstance(exponent, (int, float))
            or not math.isfinite(exponent) or exponent < 0
            or isinstance(maximum, bool) or not isinstance(maximum, (int, float))
            or not math.isfinite(maximum) or maximum < 1
            or not isinstance(source_weights, dict)):
        raise ValueError("Balance exponent must be finite/nonnegative and max_class_weight at least one")
    for name, value in source_weights.items():
        if (not isinstance(name, str) or not name.strip() or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError("Balance source weights require nonempty names and finite positive values")
    coarse = _probability_model(settings.get("probability_model", "flat")) == "coarse-factorized"
    targets = np.asarray(targets, dtype=np.float64)
    partial = np.asarray(partial, dtype=bool)
    if targets.shape != (len(items), 4 if coarse else len(CHOICES)) or partial.shape != (len(items),):
        raise ValueError("Balance weights need one normalized target and partial flag per TRAIN item")
    if not np.isfinite(targets).all() or (targets < 0).any() or not np.allclose(targets.sum(axis=1), 1):
        raise ValueError("Balance weights need finite normalized nonnegative TRAIN targets")
    groups, sources = {}, {}
    for index, item in enumerate(items):
        source = item.get("training_source") or item.get("source") or "primary"
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Training source must be a nonempty string")
        groups.setdefault((source, False if coarse else bool(partial[index])), []).append(index)
        sources.setdefault(source, []).append(index)
    weights, details = np.ones(len(items), dtype=np.float64), []
    for (source, grouped), indices in sorted(groups.items()):
        values = targets[indices]
        if grouped:
            values = np.column_stack((values[:, :2].sum(axis=1), values[:, 2],
                                      values[:, 3:OFF_TOPIC].sum(axis=1), values[:, OFF_TOPIC]))
        masses = values.sum(axis=0)
        present = masses > 0
        factors = np.zeros(len(masses), dtype=np.float64)
        factors[present] = np.minimum((masses[present].mean() / masses[present]) ** exponent, maximum)
        weights[indices] = values @ factors
        details.append({"source": source, "partial_labels": grouped, "items": len(indices),
                        "class_order": (["negative", "neutral", "positive", "irrelevant"] if coarse else
                                        ["A+B", "C", "D+E", "F"] if grouped else list(CHOICES)),
                        "class_mass": masses.tolist(), "class_factors": factors.tolist()})
    normalization = {}
    for source, indices in sorted(sources.items()):
        mean = float(weights[indices].mean())
        source_weight = float(source_weights.get(source, 1))
        weights[indices] *= source_weight / mean
        normalization[source] = {"items": len(indices), "mean_before_normalization": mean,
                                 "source_weight": source_weight}
    weights = weights.astype(np.float32)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Balance weights must remain finite and positive at float32 precision")
    return weights, {"format_version": "train-source-soft-class-balance-v1",
                     "exponent": float(exponent), "max_class_weight": float(maximum),
                     "groups": details, "source_normalization": normalization,
                     "mass_reference": "Normalized TRAIN targets only; confidence not included",
                     "normalization": "Class factors clipped first; example factors mean-normalized across each source, then source_weight applied",
                     "weights": _weight_summary(weights)}


def _snapshot(model, bundle, lora, entity, aliases, training, statistic, encoder_state=None,
              probability_model="flat"):
    artifact = {
        "format_version": 3 if encoder_state is not None else 2, "kind": "encoder", "input_format": INPUT_FORMAT,
        "decision_rule": "marginal-relevance-p-f-below-half-v1", "score_statistic": statistic,
        "entity": entity, "aliases": list(aliases), "encoder": copy.deepcopy(bundle), "lora": dict(lora),
        "state": _state(model), "training": copy.deepcopy(training),
    }
    if encoder_state is not None:
        artifact["encoder_state"] = copy.deepcopy(encoder_state)
    if _probability_model(probability_model) != "flat":
        artifact["probability_model"] = probability_model
    return artifact


def _optimizer_groups(named, learning_rate, head_learning_rate, pretrained_head_learning_rate=None, encoder_learning_rate=None):
    """Group trainable weights; an omitted pretrained rate keeps the old groups."""
    pretrained_rate = head_learning_rate if pretrained_head_learning_rate is None else float(pretrained_head_learning_rate)
    groups = [{"params": [], "lr": learning_rate},
              {"params": [], "lr": head_learning_rate},
              {"params": [], "lr": pretrained_rate}]
    shared_head = []
    encoder = []
    for name, parameter in named:
        if ".lora_" in name:
            groups[0]["params"].append(parameter)
        elif name.startswith(ENCODER_PREFIX):
            encoder.append(parameter)
        else:
            shared_head.append(parameter)
            index = 1 if "classifier" in name.split(".") else 2
            groups[index]["params"].append(parameter)
    if pretrained_rate == head_learning_rate:
        groups = [groups[0], {"params": shared_head, "lr": head_learning_rate}]
    if encoder:
        groups.append({"params": encoder, "lr": pretrained_rate if encoder_learning_rate is None else float(encoder_learning_rate)})
    return groups


def fit(items, targets, config, prior=None, on_epoch=None):
    global _RUNTIME
    settings = config["student_encoder"]
    bundle = prepare(config)
    entity = str(config.get("entity") or "").strip()
    aliases = list(config.get("aliases", []))
    probability_model = _probability_model(settings.get("probability_model", "flat"))
    targets, partial, sample_weights, relevance_only = _training_targets(items, targets, probability_model)
    dropout = settings.get("dropout")
    _dropout_overrides(dropout)
    balance_weights, balance_metadata = _balance_weights(items, targets, partial, settings)
    if balance_metadata is not None:
        balance_metadata["effective_weights"] = _weight_summary(sample_weights * balance_weights)
        balance_metadata["confidence_weights"] = _weight_summary(sample_weights)
    lora = {"rank": int(settings.get("lora_rank", 8)), "alpha": int(settings.get("lora_alpha", 16)),
            "dropout": float(settings.get("lora_dropout", 0.05))}
    if not entity or bundle["max_length"] < 16 or bundle["batch_size"] < 1 or lora["rank"] < 1 or lora["alpha"] < 1 or not 0 <= lora["dropout"] < 1:
        raise ValueError("Student requires an entity and valid input, batch and LoRA settings")
    if prior is not None and (prior.get("kind") != "encoder" or prior.get("input_format") != INPUT_FORMAT
                              or _base_identity(prior["encoder"]) != _base_identity(bundle)
                              or prior["entity"] != entity or prior["lora"] != lora
                              or prior.get("probability_model", "flat") != probability_model):
        raise ValueError("Student base, entity or LoRA architecture changed; train from scratch")
    epochs = int(settings.get("incremental_epochs", 1) if prior is not None else settings.get("epochs", 3))
    accumulation = int(settings.get("gradient_accumulation", 4))
    max_steps = settings.get("max_steps")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("Student max_steps must be null or a positive integer")
    temperature = float(settings.get("distillation_temperature", 1.0))
    if epochs < 1 or accumulation < 1 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Student epochs, accumulation and distillation temperature must be positive")
    unfreeze = settings.get("unfreeze_last_n_layers", 0)
    train_lora = settings.get("train_lora", True)
    amp_dtype = settings.get("amp_dtype")
    if type(unfreeze) is not int or unfreeze < 0:
        raise ValueError("unfreeze_last_n_layers must be a nonnegative integer")
    if type(train_lora) is not bool:
        raise ValueError("train_lora must be a boolean")
    if amp_dtype not in {None, "bfloat16"} or amp_dtype is not None and bundle["dtype"] != "float32":
        raise ValueError("amp_dtype must be null or bfloat16 with float32 model parameters")
    started = time.perf_counter()
    with MODEL_LOCK:
        import torch

        seed = int(settings.get("seed", 42))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        _RUNTIME = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        encoder_state = copy.deepcopy(prior.get("encoder_state")) if prior is not None else None
        model, tokenizer, device = _load(bundle, lora, prior["state"] if prior is not None else None,
                                         encoder_state=encoder_state, unfreeze_last_n_layers=unfreeze, dropout=dropout,
                                         probability_model=probability_model)
        if not train_lora:
            for name, parameter in model.named_parameters():
                if ".lora_" in name:
                    parameter.requires_grad_(False)
        device_type = torch.device(device).type
        if amp_dtype is not None and (device_type not in {"cpu", "cuda"}
                                      or device_type == "cuda" and not torch.cuda.is_bf16_supported()):
            raise ValueError("bfloat16 autocast requires a supported CPU or CUDA device")
        if settings.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        unfrozen = {name: parameter for name, parameter in _encoder_parameters(model).items() if parameter.requires_grad}
        encoder_keys = sorted(set(unfrozen) | set(encoder_state["keys"] if encoder_state is not None else []))
        encoder_trainable = sum(parameter.numel() for parameter in unfrozen.values())
        learning_rate = float(settings.get("learning_rate", 0.0003))
        head_learning_rate = float(settings.get("head_learning_rate", 0.001))
        pretrained_head_learning_rate = settings.get("pretrained_head_learning_rate")
        pretrained_head_learning_rate = (head_learning_rate if pretrained_head_learning_rate is None
                                         else float(pretrained_head_learning_rate))
        encoder_learning_rate = float(settings.get("encoder_learning_rate", 0.00001))
        optimizer = torch.optim.AdamW(
            _optimizer_groups(named, learning_rate, head_learning_rate, pretrained_head_learning_rate, encoder_learning_rate),
            weight_decay=float(settings.get("weight_decay", 0.01)),
        )
        LOGGER.info("Student encoder: %d unfrozen blocks, %d trainable backbone parameters, AMP %s",
                    unfreeze, encoder_trainable, amp_dtype or "disabled")
        model.train()
        total_loss, batches, steps, seen, completed_epochs = 0.0, 0, 0, 0, 0
        count = math.ceil(len(items) / bundle["batch_size"])
        planned_steps = epochs * math.ceil(count / accumulation)
        if max_steps is not None:
            planned_steps = min(planned_steps, max_steps)
        log_every = int(settings.get("log_every_steps", 50))
        training_started = time.perf_counter()
        training = {}
        for epoch in range(epochs):
            order = rng.permutation(len(items))
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            for batch_index, start in enumerate(range(0, len(items), bundle["batch_size"])):
                selected = order[start:start + bundle["batch_size"]]
                batch = _batch(tokenizer, [items[int(index)] for index in selected], entity, aliases, bundle, device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16) if amp_dtype is not None else nullcontext():
                    logits = model(**batch).logits
                    expected = torch.as_tensor(targets[selected], device=device, dtype=torch.float32)
                    selected_partial = torch.as_tensor(partial[selected], device=device, dtype=torch.bool)
                    selected_weights = torch.as_tensor(sample_weights[selected], device=device, dtype=torch.float32)
                    selected_balance = (torch.as_tensor(balance_weights[selected], device=device, dtype=torch.float32)
                                        if balance_metadata is not None else None)
                    selected_relevance_only = torch.as_tensor(relevance_only[selected], device=device, dtype=torch.bool)
                    loss = _loss(logits, expected, selected_partial, temperature, torch, selected_weights, selected_balance,
                                 probability_model, selected_relevance_only)
                if not torch.isfinite(loss):
                    raise ValueError("Student training produced a non-finite loss")
                # Weight a short final batch by its actual number of examples.
                group_start = (batch_index // accumulation) * accumulation * bundle["batch_size"]
                group_items = min(accumulation * bundle["batch_size"], len(items) - group_start)
                (loss * len(selected) / group_items).backward()
                value = float(loss.detach().cpu())
                epoch_loss += value
                total_loss += value
                batches += 1
                seen += len(selected)
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == count:
                    torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named], float(settings.get("max_grad_norm", 1.0)))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
                    if steps == 1 or log_every > 0 and steps % log_every == 0:
                        elapsed = time.perf_counter() - training_started
                        LOGGER.info("Student step %d/%d: KL %.5f, examples %d, elapsed %.1fs, ETA %.1fs",
                                    steps, planned_steps, value, seen, elapsed, elapsed / steps * (planned_steps - steps))
                    if max_steps is not None and steps >= max_steps:
                        break
            if batch_index + 1 == count:
                completed_epochs += 1
            LOGGER.info("Student epoch %d/%d: KL %.5f, optimizer steps %d", epoch + 1, epochs, epoch_loss / (batch_index + 1), steps)
            finished = epoch + 1 == epochs or max_steps is not None and steps >= max_steps
            if unfrozen and (on_epoch is not None or finished):
                encoder_state = _save_encoder(model, bundle, encoder_keys)
            training = {"max_steps": max_steps, "optimizer_steps": steps, "examples_seen": seen,
                        "epochs_requested": epochs, "epochs_completed": completed_epochs,
                        "seed": seed, "gradient_accumulation": accumulation,
                        "distillation_temperature": temperature,
                        "sample_weight_sum": float(sample_weights.sum()),
                        "sample_weighted_items": int((sample_weights != 1).sum()),
                        "partial_label_items": int(partial.sum()),
                        "relevance_only_items": int(relevance_only.sum()),
                        "gradient_checkpointing": bool(settings.get("gradient_checkpointing", False)),
                        "unfreeze_last_n_layers": unfreeze,
                        "unfrozen_encoder_layers": sorted({_encoder_layer(name) for name in unfrozen}),
                        "encoder_trainable_parameters": encoder_trainable, "base_encoder_frozen": not bool(unfrozen),
                        "encoder_override_bytes": encoder_state["asset"]["size_bytes"] if encoder_state is not None else 0,
                        "amp_dtype": amp_dtype,
                        "optimizer": {"name": "AdamW", "state_restored": False,
                                      "learning_rate": learning_rate,
                                      "head_learning_rate": head_learning_rate,
                                      "pretrained_head_learning_rate": pretrained_head_learning_rate,
                                      "encoder_learning_rate": encoder_learning_rate if unfrozen else None,
                                      "weight_decay": float(settings.get("weight_decay", 0.01)),
                                      "max_grad_norm": float(settings.get("max_grad_norm", 1.0))}}
            if dropout is not None:
                training["dropout"] = float(dropout)
            if balance_metadata is not None:
                training["training_balance"] = copy.deepcopy(balance_metadata)
            if probability_model != "flat":
                training["probability_model"] = probability_model
            if on_epoch is not None:
                model.eval()
                checkpoint = _snapshot(model, bundle, lora, entity, aliases, training, config.get("score_statistic", "mean"),
                                       encoder_state, probability_model)
                _RUNTIME = (_identity(checkpoint), model, tokenizer, device)
                try:
                    on_epoch(checkpoint, {"epoch": epoch + 1, "objective": epoch_loss / (batch_index + 1),
                                          "elapsed_seconds": time.perf_counter() - started, **training})
                finally:
                    # The next update must not mutate an inference cache identified
                    # by the previous epoch's immutable adapter snapshot.
                    _RUNTIME = None
                    model.train()
            if max_steps is not None and steps >= max_steps:
                break
        model.eval()
        artifact = _snapshot(model, bundle, lora, entity, aliases, training, config.get("score_statistic", "mean"),
                             encoder_state, probability_model)
        _RUNTIME = (_identity(artifact), model, tokenizer, device)
        optimization = {
            "trainable_parameters": sum(parameter.numel() for _, parameter in named),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "adapter_bytes": sum(value.nbytes for value in artifact["state"].values()),
            "epochs": completed_epochs, "epochs_requested": epochs, "max_steps": max_steps,
            "iterations": steps, "examples_seen": seen, "objective": total_loss / batches,
            "warm_started": prior is not None, "base_encoder_frozen": not bool(unfrozen),
            "training_method": ("Three-class conditional tone and learned relevance; confidence-weighted joint KL"
                                if probability_model == "coarse-factorized" else
                                "LoRA, classification head and configured encoder blocks; confidence-weighted KL and partial four-class marginal loss"),
            "unfreeze_last_n_layers": unfreeze, "unfrozen_encoder_layers": training["unfrozen_encoder_layers"],
            "encoder_trainable_parameters": encoder_trainable,
            "encoder_override_bytes": training["encoder_override_bytes"], "amp_dtype": amp_dtype,
            "encoder_sha256": bundle["files"]["model.safetensors"]["sha256"],
            "elapsed_seconds": time.perf_counter() - started, "device": device,
            "optimizer": copy.deepcopy(training["optimizer"]),
            "partial_label_items": int(partial.sum()),
            "relevance_only_items": int(relevance_only.sum()),
            "polarity_only_items": sum(bool(item.get("polarity_only")) for item in items),
            "sample_weight_sum": float(sample_weights.sum()),
            "sample_weighted_items": int((sample_weights != 1).sum()),
        }
        for name in ("dropout", "training_balance", "probability_model"):
            if name in training:
                optimization[name] = copy.deepcopy(training[name])
        return artifact, optimization


def predict(artifact, items):
    global _RUNTIME
    if artifact.get("input_format") != INPUT_FORMAT:
        raise ValueError("Unknown student input format")
    probability_model = _probability_model(artifact.get("probability_model", "flat"))
    with MODEL_LOCK:
        import torch

        identity = _identity(artifact)
        if _RUNTIME is None or _RUNTIME[0] != identity:
            _RUNTIME = None
            gc.collect()
            model, tokenizer, device = _load(artifact["encoder"], artifact["lora"], artifact["state"],
                                             encoder_state=artifact.get("encoder_state"), probability_model=probability_model)
            model.eval()
            _RUNTIME = (identity, model, tokenizer, device)
        _, model, tokenizer, device = _RUNTIME
        values = []
        size = artifact["encoder"]["batch_size"]
        with torch.inference_mode():
            for start in range(0, len(items), size):
                batch = _batch(tokenizer, items[start:start + size], artifact["entity"], artifact["aliases"], artifact["encoder"], device)
                logits = model(**batch).logits
                if probability_model == "flat":
                    probabilities = torch.softmax(logits.float(), dim=-1)
                else:
                    coarse = _log_probabilities(logits, torch, probability_model).exp()
                    probabilities = coarse.new_zeros((len(coarse), len(CHOICES)))
                    probabilities[:, [0, 2, 4, OFF_TOPIC]] = coarse
                values.append(probabilities.cpu().numpy())
        probabilities = np.concatenate(values).astype(np.float64)
        if not np.isfinite(probabilities).all():
            raise ValueError("Student inference produced non-finite probabilities")
        return probabilities
