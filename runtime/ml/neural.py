"""ModernBERT LoRA training. Revisions contain adapters, not copies of the base."""

import copy
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import urljoin, urlsplit

import numpy as np

from lib.model_assets import asset_spec, ensure_asset, file_sha256
from lib.sentiment import CHOICES, OFF_TOPIC


LOGGER = logging.getLogger(__name__)
MODEL_LOCK = threading.RLock()
INPUT_FORMAT = "target-comment-parent-title-v1"
COMPANIONS = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
_RUNTIME = None


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


def _load(bundle, lora, state=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    torch.set_num_threads(bundle["threads"])
    directory = _model_directory(bundle)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        directory, num_labels=len(CHOICES), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="sdpa", reference_compile=False,
    )
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
    digest = hashlib.sha256(json.dumps({
        "base": _base_identity(artifact["encoder"]), "lora": artifact["lora"],
        "entity": artifact["entity"], "aliases": artifact["aliases"],
        "input_format": artifact["input_format"], "training": artifact.get("training"), "runtime": {
            name: artifact["encoder"][name] for name in ("device", "dtype", "max_length", "threads")
        },
    }, sort_keys=True).encode())
    for name, value in sorted(artifact["state"].items()):
        digest.update(name.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _loss(logits, expected, partial, temperature, torch):
    log_probabilities = torch.log_softmax(logits.float() / temperature, dim=-1)
    if temperature != 1.0:
        expected = expected.pow(1.0 / temperature)
        expected = expected / expected.sum(dim=-1, keepdim=True)
    losses = torch.nn.functional.kl_div(log_probabilities, expected, reduction="none").sum(dim=-1)
    if partial.any():
        grouped = torch.stack([
            torch.logsumexp(log_probabilities[:, :2], dim=-1), log_probabilities[:, 2],
            torch.logsumexp(log_probabilities[:, 3:OFF_TOPIC], dim=-1),
        ], dim=-1)
        reference = torch.stack([expected[:, :2].sum(dim=-1), expected[:, 2], expected[:, 3:OFF_TOPIC].sum(dim=-1)], dim=-1)
        marginal = torch.nn.functional.kl_div(grouped, reference, reduction="none").sum(dim=-1)
        losses = torch.where(partial, marginal, losses)
    return losses.mean() * temperature ** 2


def _snapshot(model, bundle, lora, entity, aliases, training, statistic):
    return {
        "format_version": 2, "kind": "encoder", "input_format": INPUT_FORMAT,
        "decision_rule": "marginal-relevance-p-f-below-half-v1", "score_statistic": statistic,
        "entity": entity, "aliases": list(aliases), "encoder": copy.deepcopy(bundle), "lora": dict(lora),
        "state": _state(model), "training": copy.deepcopy(training),
    }


def fit(items, targets, config, prior=None, on_epoch=None):
    global _RUNTIME
    settings = config["student_encoder"]
    bundle = prepare(config)
    entity = str(config.get("entity") or "").strip()
    aliases = list(config.get("aliases", []))
    targets = np.asarray(targets, dtype=np.float32)
    if targets.shape != (len(items), len(CHOICES)) or not len(items) or not np.isfinite(targets).all() or (targets < 0).any() or (targets.sum(axis=1) <= 0).any():
        raise ValueError("Training needs finite nonnegative six-class targets")
    targets = targets / targets.sum(axis=1, keepdims=True)
    partial = np.asarray([bool(item.get("polarity_only")) for item in items])
    if (targets[partial, OFF_TOPIC] > 1e-6).any():
        raise ValueError("Polarity-only references must refer to a relevant target")
    lora = {"rank": int(settings.get("lora_rank", 8)), "alpha": int(settings.get("lora_alpha", 16)),
            "dropout": float(settings.get("lora_dropout", 0.05))}
    if not entity or bundle["max_length"] < 16 or bundle["batch_size"] < 1 or lora["rank"] < 1 or lora["alpha"] < 1 or not 0 <= lora["dropout"] < 1:
        raise ValueError("Student requires an entity and valid input, batch and LoRA settings")
    if prior is not None and (prior.get("kind") != "encoder" or prior.get("input_format") != INPUT_FORMAT
                              or _base_identity(prior["encoder"]) != _base_identity(bundle)
                              or prior["entity"] != entity or prior["lora"] != lora):
        raise ValueError("Student base, entity or LoRA architecture changed; train from scratch")
    epochs = int(settings.get("incremental_epochs", 1) if prior is not None else settings.get("epochs", 3))
    accumulation = int(settings.get("gradient_accumulation", 4))
    max_steps = settings.get("max_steps")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("Student max_steps must be null or a positive integer")
    temperature = float(settings.get("distillation_temperature", 1.0))
    if epochs < 1 or accumulation < 1 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Student epochs, accumulation and distillation temperature must be positive")
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
        model, tokenizer, device = _load(bundle, lora, prior["state"] if prior is not None else None)
        if settings.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        adapter_parameters = [parameter for name, parameter in named if ".lora_" in name]
        head_parameters = [parameter for name, parameter in named if ".lora_" not in name]
        learning_rate = float(settings.get("learning_rate", 0.0003))
        optimizer = torch.optim.AdamW([
            {"params": adapter_parameters, "lr": learning_rate},
            {"params": head_parameters, "lr": float(settings.get("head_learning_rate", 0.001))},
        ], weight_decay=float(settings.get("weight_decay", 0.01)))
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
                logits = model(**batch).logits
                expected = torch.as_tensor(targets[selected], device=device, dtype=torch.float32)
                selected_partial = torch.as_tensor(partial[selected], device=device, dtype=torch.bool)
                loss = _loss(logits, expected, selected_partial, temperature, torch)
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
            training = {"max_steps": max_steps, "optimizer_steps": steps, "examples_seen": seen,
                        "epochs_requested": epochs, "epochs_completed": completed_epochs,
                        "seed": seed, "gradient_accumulation": accumulation,
                        "distillation_temperature": temperature,
                        "gradient_checkpointing": bool(settings.get("gradient_checkpointing", False)),
                        "optimizer": {"name": "AdamW", "state_restored": False,
                                      "learning_rate": learning_rate,
                                      "head_learning_rate": float(settings.get("head_learning_rate", 0.001)),
                                      "weight_decay": float(settings.get("weight_decay", 0.01)),
                                      "max_grad_norm": float(settings.get("max_grad_norm", 1.0))}}
            if on_epoch is not None:
                model.eval()
                checkpoint = _snapshot(model, bundle, lora, entity, aliases, training, config.get("score_statistic", "mean"))
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
        artifact = _snapshot(model, bundle, lora, entity, aliases, training, config.get("score_statistic", "mean"))
        _RUNTIME = (_identity(artifact), model, tokenizer, device)
        optimization = {
            "trainable_parameters": sum(parameter.numel() for _, parameter in named),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "adapter_bytes": sum(value.nbytes for value in artifact["state"].values()),
            "epochs": completed_epochs, "epochs_requested": epochs, "max_steps": max_steps,
            "iterations": steps, "examples_seen": seen, "objective": total_loss / batches,
            "warm_started": prior is not None, "base_encoder_frozen": True,
            "training_method": "LoRA attention and classification head; KL distillation and marginal loss for polarity-only references",
            "encoder_sha256": bundle["files"]["model.safetensors"]["sha256"],
            "elapsed_seconds": time.perf_counter() - started, "device": device,
            "polarity_only_items": int(partial.sum()),
        }
        return artifact, optimization


def predict(artifact, items):
    global _RUNTIME
    if artifact.get("input_format") != INPUT_FORMAT:
        raise ValueError("Unknown student input format")
    with MODEL_LOCK:
        import torch

        identity = _identity(artifact)
        if _RUNTIME is None or _RUNTIME[0] != identity:
            _RUNTIME = None
            gc.collect()
            model, tokenizer, device = _load(artifact["encoder"], artifact["lora"], artifact["state"])
            model.eval()
            _RUNTIME = (identity, model, tokenizer, device)
        _, model, tokenizer, device = _RUNTIME
        values = []
        size = artifact["encoder"]["batch_size"]
        with torch.inference_mode():
            for start in range(0, len(items), size):
                batch = _batch(tokenizer, items[start:start + size], artifact["entity"], artifact["aliases"], artifact["encoder"], device)
                values.append(torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy())
        probabilities = np.concatenate(values).astype(np.float64)
        if not np.isfinite(probabilities).all():
            raise ValueError("Student inference produced non-finite probabilities")
        return probabilities
