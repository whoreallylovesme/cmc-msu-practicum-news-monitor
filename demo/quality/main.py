"""Подготовка независимых данных и оценка учителя; все параметры в YAML."""

import logging
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import threading


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.cli import configure_logging, load_cli_config, write_json
from data import prepare, read_split
from metrics import label_with, save_result


LOGGER = logging.getLogger(__name__)
DATASETS = {"newsmtsc_rw", "newsmtsc_mt", "weather_sentiment", "airline_sentiment"}


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@contextmanager
def teacher(config):
    sys.path.insert(0, str(ROOT / "runtime/teacher"))
    from model import TeacherRuntime, load_settings
    from lib.teacher_contracts import TeacherComment

    settings = load_settings(local_path(config["teacher_config"]))
    runtime = None
    initialization = threading.Lock()

    def predict(entity, item, aliases=()):
        nonlocal runtime
        if runtime is None:
            with initialization:
                if runtime is None:
                    runtime = TeacherRuntime(settings)
        return runtime.distribution(entity, aliases, TeacherComment(**item))

    try:
        yield settings, predict
    finally:
        if runtime is not None:
            runtime.close()


def evaluation_settings(config):
    selected, split = config["datasets"], config.get("split", "dev")
    assert isinstance(selected, list) and selected and set(selected) <= DATASETS
    assert len(selected) == len(set(selected)), "Не повторяйте один dataset в списке"
    assert split in {"dev", "test"}, "Для независимой оценки используйте dev или test"
    mark_target = config.get("mark_target", True)
    assert type(mark_target) is bool
    return selected, split, mark_target


def evaluate(config, directory, output):
    selected, split, mark_target = evaluation_settings(config)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with teacher(config) as (settings, predict):
        for dataset in selected:
            rows = read_split(directory, dataset, split)
            predictions = label_with(predict, rows, output / "predictions.sqlite3", settings["revision"],
                                     mark_target=mark_target, protocol=settings["protocol"])
            path = output / f"{dataset}_{split}_{settings['revision']}_{predictions[0]['protocol_sha256'][:12]}.json"
            report = save_result(path, rows, predictions, int(config.get("bootstrap_samples", 1000)))
            report["model_protocol"] = settings["protocol"]
            report["timing"]["callback_seconds_including_lazy_load"] = report["timing"].pop("warm_inference_seconds")
            report["timing"]["note"] = "Only uncached callbacks; the first fresh callback includes lazy teacher initialization and model loading. This is not an isolated backbone benchmark."
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            LOGGER.info("%s/%s: polarity F1=%s; relevance F1=%s; результат %s", dataset, split,
                        report["metrics"]["polarity_macro_f1"], report["metrics"]["relevance_positive_f1"], path)
            results.append({"dataset": dataset, "split": split, "result": str(path), "metrics": report["metrics"]})
    return results


def main():
    configure_logging()
    _, config = load_cli_config(Path(__file__).with_name("config.yaml"))
    operation = config.get("operation", "prepare")
    assert operation in {"prepare", "evaluate_teacher", "evaluate_student", "label_demo"}, "operation: prepare, evaluate_teacher, evaluate_student или label_demo"
    directory = local_path(config["data_dir"])
    if operation == "prepare":
        manifest = prepare(directory)
        write_json({"manifest": str(directory / "manifest.json"), "datasets": [row["name"] for row in manifest["datasets"]]})
    elif operation == "evaluate_teacher":
        write_json({"results": evaluate(config, directory, local_path(config["output_dir"]))})
    elif operation == "evaluate_student":
        from student_evaluation import evaluate_student
        evaluation_settings(config)
        write_json({"results": evaluate_student(config, directory, local_path(config["output_dir"]))})
    else:
        from label_demo import label_snapshot
        with teacher(config) as (settings, predict):
            write_json(label_snapshot(config, settings, predict, local_path(config["output_dir"])))


if __name__ == "__main__":
    main()
