"""Оценить сохранённую версию ученика без обучения и передачи gold в модель."""

import copy
from importlib import metadata
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime" / "ml"))

from runtime.common.hashing import file_sha256
from registry import Registry
import student


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_student(config):
    settings = config["student"]
    source = local_path(settings["registry_path"])
    assert source.is_file(), f"Нет готового registry {source}; сначала обучите модель через ML API"
    registry = Registry(source)
    models = registry.state()["models"]
    model_id = settings.get("model_id")
    if model_id is None:
        entity = settings.get("entity", "ChatGPT")
        candidates = [model for model in models.values() if model.get("version")
                      and model.get("entity", "").casefold() == entity.casefold()]
        assert candidates, f"Нет обученной модели для {entity}"

        def trained_at(model):
            return next((row["trained_at"] for row in model["versions"] if row["version"] == model["version"]), "")

        model_id = max(candidates, key=lambda model: (trained_at(model), model["id"]))["id"]
    model = models[model_id]
    assert model.get("version"), "Выбранная модель ещё не обучена"
    version = settings.get("version")
    version = model["version"] if version is None else version
    assert type(version) is int and version > 0
    artifact = copy.deepcopy(registry.load(model_id, version))
    with sqlite3.connect(source) as connection:
        checksum = connection.execute("SELECT sha256 FROM model_artifacts WHERE model_id=? AND version=?",
                                      (model_id, version)).fetchone()[0]
    runtime = artifact.get("encoder")
    if not isinstance(runtime, dict):
        runtime = artifact
    device = settings.get("device")
    if device is not None:
        assert device in {"auto", "cpu", "mps", "cuda"}
        runtime["device"] = device
    if settings.get("cache_dir"):
        runtime["cache_dir"] = str(local_path(settings["cache_dir"]))
    source_names = ["runtime/ml/student.py", "runtime/ml/neural.py", "lib/supervision.py",
                    "demo/quality/student_evaluation.py"]
    if runtime is artifact:
        source_names.append("runtime/ml/backends/causal_lm_ensemble.py")
        source_names.extend(("runtime/prompts/evidence_system.txt", "runtime/prompts/label_only_system.txt"))
    protocol = {
        "kind": "trained-student", "model_id": model_id, "version": version,
        "artifact_sha256": checksum, "entity": artifact["entity"],
        "input_format": artifact.get("input_format"), "decision_rule": artifact.get("decision_rule"),
        "score_statistic": artifact["score_statistic"], "device": runtime.get("device", "auto"),
        "source_sha256": {name: file_sha256(ROOT / name) for name in source_names},
        "libraries": {name: metadata.version(name) for name in ("torch", "transformers", "peft")},
    }

    def predict(entity, item, aliases=()):
        # Benchmark annotations remain outside this allowlisted input projection.
        projected = {name: item.get(name) for name in ("id", "story_id", "text", "story_title", "parent_text", "parent_id")}
        projected["entity"] = entity
        return student.predict(artifact, [projected])[0]

    return protocol, predict


def evaluate_student(config, directory, output):
    from data import read_split
    from metrics import label_with, save_result

    protocol, predict = load_student(config)
    revision = "student-" + protocol["artifact_sha256"]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in config["datasets"]:
        rows = read_split(directory, dataset, config.get("split", "dev"))
        predictions = label_with(predict, rows, output / "predictions.sqlite3", revision,
                                 mark_target=config.get("mark_target", True), protocol=protocol)
        path = output / f"{dataset}_{rows[0]['split']}_{revision}_{predictions[0]['protocol_sha256'][:12]}.json"
        report = save_result(path, rows, predictions, int(config.get("bootstrap_samples", 1000)))
        report["model_protocol"] = protocol
        report["timing"]["callback_seconds_including_lazy_load"] = report["timing"].pop("warm_inference_seconds")
        report["timing"]["note"] = "Only uncached callbacks; the first fresh callback includes lazy model construction and loading. This is not an isolated backbone benchmark."
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        results.append({"dataset": dataset, "split": rows[0]["split"], "result": str(path),
                        "model_id": protocol["model_id"], "version": protocol["version"], "metrics": report["metrics"]})
    return results
