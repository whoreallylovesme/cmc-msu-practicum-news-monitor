"""Freeze teacher distributions for the bundled offline Hacker News snapshot."""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime" / "ml"))
from lib.config import load_config, load_monitoring, load_project_config
from teacher import LlamaTeacher, cache_namespace, ensure_model, input_fingerprint, llama_teacher, mention_policy


def label_comments(comments, entity, config, base_url, workers, save_result):
    """Keep at most one request per worker in flight and save on the caller thread."""
    local = threading.local()
    clients = []
    clients_lock = threading.Lock()

    def classify(comment):
        if not hasattr(local, "teacher"):
            local.teacher = LlamaTeacher(base_url, config)
            with clients_lock:
                clients.append(local.teacher)
        started = time.perf_counter()
        probabilities = local.teacher.distribution(entity, comment)
        return {
            "comment_id": str(comment["id"]),
            "input_hash": input_fingerprint(entity, comment),
            "probabilities": probabilities,
            "evidence": local.teacher.last_evidence,
            "mention_kind": getattr(local.teacher, "last_mention_kind", None),
            "inference_ms": (time.perf_counter() - started) * 1000,
        }

    remaining = iter(comments)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="snapshot-teacher")
    pending = set()

    def fill_slots():
        while len(pending) < workers:
            comment = next(remaining, None)
            if comment is None:
                break
            pending.add(executor.submit(classify, comment))

    try:
        fill_slots()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                # Keep failed futures in pending so every completed result is
                # drained below even if a different worker fails first.
                result = future.result()
                pending.remove(future)
                save_result(result)
            fill_slots()
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    finally:
        # Running requests finish before their sessions or the shared server
        # close. Preserve their successes for a subsequent resumable invocation.
        executor.shutdown(wait=True, cancel_futures=True)
        try:
            for future in pending:
                if future.cancelled():
                    continue
                try:
                    result = future.result()
                except BaseException:
                    continue
                save_result(result)
        finally:
            for client in clients:
                client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(load_project_config()["path_to_offline_demo_data"]))
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "ml" / "hn_snapshot_labels.json")
    parser.add_argument("--entity", help="Override monitoring.query from the project configuration")
    parser.add_argument("--weights", type=Path, help="Use a compatible local GGUF")
    parser.add_argument("--teacher-name", help="Model name to display in the snapshot provenance")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".local-data" / "ml" / "weights")
    parser.add_argument("--llama-binary", default=os.environ.get("ML_LLAMA_BINARY", "llama-server"))
    parser.add_argument("--gpu-layers", type=int, help="Override llama_gpu_layers, e.g. 99 for native Metal")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1,
                        help="Concurrent clients sharing one llama-server (1–4; default: 1)")
    args = parser.parse_args()
    entity = args.entity or load_monitoring()["query"]
    config = load_config(ROOT / "runtime" / "ml" / "config.yaml", env_var="ML_CONFIG")
    config["monitoring"] = load_monitoring()
    configured_source = config["model_weights"]
    config.update(model_cache_dir=str(args.cache_dir.resolve()), llama_binary=args.llama_binary)
    # llama.cpp divides the total context among its slots. Keep the configured
    # context available to every request when enabling parallel labeling.
    config["llama_parallel"] = args.workers
    config["llama_context"] *= args.workers
    if args.weights is not None:
        config["model_weights"] = {"path": str(args.weights.resolve()), "is_local": True}
    if args.gpu_layers is not None:
        config["llama_gpu_layers"] = args.gpu_layers
    digest = ensure_model(config)
    if digest is None:
        raise ValueError("Specify weights to freeze a snapshot with verifiable provenance")

    stories = json.loads(args.dataset.read_text(encoding="utf-8"))
    comments = [comment for story in stories for comment in story["comments"]]
    origin = config["model_weights"]
    if digest == config.get("model_checksums", {}).get(configured_source["path"]):
        origin = configured_source
    origin_name = Path(origin["path"] if origin["is_local"] else urlsplit(origin["path"]).path).stem
    output = {
        "entity": entity,
        "teacher": {
            "model": args.teacher_name or origin_name,
            "revision": (config.get("teacher_revision") or "sha256:" + digest)
            if origin["is_local"] else origin["path"],
            "sha256": digest,
            "inference_namespace": cache_namespace(config, "live", digest),
            "inference_mode": config.get("teacher_inference_mode", "reasoning"),
            "probability_kind": "label_smoothed_hard_predictions" if config.get("teacher_inference_mode") == "structured" else "conditional_token_probabilities",
            "mention_policy": mention_policy(config, entity),
        },
        "items": [],
    }
    progress = args.output.with_suffix(".progress.json")
    if progress.is_file():
        previous = json.loads(progress.read_text(encoding="utf-8"))
        if previous.get("entity") != entity or previous.get("teacher") != output["teacher"]:
            raise ValueError(f"Checkpoint belongs to a different teacher or entity: {progress}")
        output["items"] = previous["items"]
    labels = {row["comment_id"]: row for row in output["items"]}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        temporary = progress.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(progress)

    remaining = [comment for comment in comments if labels.get(str(comment["id"]), {}).get("input_hash")
                 != input_fingerprint(entity, comment)]
    completed = len(comments) - len(remaining)

    def save_result(row):
        nonlocal completed
        labels[row["comment_id"]] = row
        output["items"] = [labels[str(comment["id"])] for comment in comments
                           if str(comment["id"]) in labels]
        checkpoint()
        completed += 1
        if completed % 24 == 0 or completed == len(comments):
            print(f"Teacher labeled {completed}/{len(comments)}", flush=True)

    if remaining:
        with llama_teacher(config, args.cache_dir / "llama-snapshot.log") as teacher:
            label_comments(remaining, entity, config, teacher.base_url, args.workers, save_result)
    else:
        print(f"Teacher labeled {completed}/{len(comments)} (checkpoint complete)", flush=True)

    output["items"] = [labels[str(comment["id"])] for comment in comments]
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    progress.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
