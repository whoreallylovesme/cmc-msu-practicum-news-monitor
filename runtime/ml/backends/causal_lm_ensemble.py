"""Causal-LM evidence student with a conservative four-member relevance vote."""

import gc
import json
import re
import threading
from pathlib import Path

import numpy as np

from common.hashing import directory_sha256, file_sha256
from common.inference import resolve_device
from common.prompts import load_prompt
from lib.teacher_contracts import canonical_input


LOCK = threading.RLock()
PROTOCOL = "causal-lm-evidence-four-vote-v1"
LEGACY_PROTOCOL = "qwen34-zero-evidence-four-vote-v1"
LEGACY_ADAPTER_MODES = (
    ("epoch1_one_letter", "label_only"),
    ("epoch2_one_letter", "label_only"),
    ("epoch1_evidence", "evidence"),
    ("epoch2_evidence", "evidence"),
)
PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
EVIDENCE_PROMPT = load_prompt(PROMPTS / "evidence_system.txt").strip()
LABEL_ONLY_PROMPT = load_prompt(PROMPTS / "label_only_system.txt").strip()


def parse_label(text):
    matches = re.findall(r"CLASS:\s*([A-F])", text)
    if not matches:
        raise ValueError("Generated response has no CLASS: A-F label")
    return "ABCDEF".index(matches[-1])


def combine_labels(base, auxiliary):
    base = np.asarray(base, dtype=np.int64)
    auxiliary = np.asarray(auxiliary, dtype=np.int64)
    if base.ndim != 1 or auxiliary.ndim != 2 or auxiliary.shape[0] != 4:
        raise ValueError("Expected one base label per item and four auxiliary votes")
    if auxiliary.shape[1] != len(base):
        raise ValueError("Base and auxiliary prediction counts differ")
    if ((base < 0) | (base >= 6)).any() or ((auxiliary < 0) | (auxiliary >= 6)).any():
        raise ValueError("Labels must be in the A-F range")
    result = base.copy()
    override = (base == 2) & ((auxiliary == 5).sum(axis=0) >= 2)
    result[override] = 5
    return result


def ensemble_members(artifact):
    if "members" not in artifact:
        members = []
        for name, mode in LEGACY_ADAPTER_MODES:
            member = dict(artifact["adapters"][name])
            member.update(name=name, mode=mode)
            members.append(member)
        return members
    configured = artifact["members"]
    if not isinstance(configured, (list, tuple)) or len(configured) != 4:
        raise ValueError("Causal LM ensemble requires four members")
    for member in configured:
        if not isinstance(member, dict) or member.get("mode") not in {"label_only", "evidence"}:
            raise ValueError("Each ensemble member must have mode label_only or evidence")
    return list(configured)


def projected_input(artifact, item):
    projected = canonical_input(artifact["entity"], item, artifact.get("aliases", []))
    return projected["entity"], projected["item"]


def prompt(tokenizer, artifact, entity, item, evidence):
    user = {
        "target": entity,
        "aliases": artifact.get("aliases", []),
        "story_title": item.get("story_title") or "",
        "parent_comment": item.get("parent_text") or "",
        "current_comment": item.get("text") or "",
    }
    system = EVIDENCE_PROMPT if evidence else LABEL_ONLY_PROMPT
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        tokenize=False, add_generation_prompt=True,
    )


def verify_artifact(artifact):
    if artifact.get("architecture") not in {PROTOCOL, LEGACY_PROTOCOL}:
        raise ValueError("Unsupported causal LM ensemble protocol")
    assets = artifact["assets"]
    for name, expected in artifact["base_assets"].items():
        path = Path(assets) / name
        if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"Invalid base asset {name}")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Base asset checksum mismatch {name}")
    if "members" in artifact:
        entries = [(member.get("name", f"member_{index}"), member)
                   for index, member in enumerate(ensemble_members(artifact))]
    else:
        entries = artifact["adapters"].items()
    for name, member in entries:
        path = Path(member["path"])
        if not path.is_dir() or directory_sha256(path) != member["sha256"]:
            raise ValueError(f"Adapter checksum mismatch {name}")


def load_tokenizer(artifact):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        artifact["assets"], local_files_only=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(artifact, tokenizer, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM

    device = resolve_device(artifact)
    model = AutoModelForCausalLM.from_pretrained(
        artifact["assets"], dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=True, attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model.to(device).eval()


def release(model):
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generated_labels(model, tokenizer, artifact, items, evidence):
    import torch

    labels = []
    batch_size = int(artifact.get("generation_batch_size", 8))
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            projected = [projected_input(artifact, item) for item in items[start:start + batch_size]]
            prompts = [prompt(tokenizer, artifact, entity, item, evidence) for entity, item in projected]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                max_length=int(artifact.get("max_length", 1024))).to(model.device)
            output = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=96,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(output[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            labels.extend(parse_label(text) for text in texts)
    return labels


def class_token_ids(tokenizer):
    token_ids = []
    for letter in "ABCDEF":
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Class label {letter} must encode to exactly one token")
        token_ids.append(encoded[0])
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Class label token IDs must be distinct")
    return token_ids


def logit_labels(model, tokenizer, artifact, items):
    import torch

    class_ids = class_token_ids(tokenizer)
    labels = []
    batch_size = int(artifact.get("logits_batch_size", 4))
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            projected = [projected_input(artifact, item) for item in items[start:start + batch_size]]
            prompts = [prompt(tokenizer, artifact, entity, item, False) for entity, item in projected]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                max_length=int(artifact.get("max_length", 1024))).to(model.device)
            logits = model(**encoded).logits[:, -1, :].float()
            labels.extend(logits[:, class_ids].argmax(dim=-1).cpu().tolist())
    return labels


def predict(artifact, items):
    import torch

    if not items:
        return np.empty((0, 6), dtype=np.float64)
    with LOCK:
        verify_artifact(artifact)
        members = ensemble_members(artifact)
        device = resolve_device(artifact)
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(float(artifact.get("cuda_memory_fraction", 1 / 3)), 0)
        tokenizer = load_tokenizer(artifact)
        model = load_model(artifact, tokenizer)
        base = generated_labels(model, tokenizer, artifact, items, True)
        release(model)
        auxiliary = []
        for member in members:
            model = load_model(artifact, tokenizer, member["path"])
            if member["mode"] == "label_only":
                labels = logit_labels(model, tokenizer, artifact, items)
            else:
                labels = generated_labels(model, tokenizer, artifact, items, True)
            auxiliary.append(labels)
            release(model)
        labels = combine_labels(base, auxiliary)
        return np.eye(6, dtype=np.float64)[labels]
