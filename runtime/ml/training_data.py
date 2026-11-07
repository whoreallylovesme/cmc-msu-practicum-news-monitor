"""Select fresh examples and preserve thread roles across a model's revisions."""

import hashlib
import json

from student import split_threads


def fingerprint(item):
    if item.get("content_hash"):
        raw_hash = item["content_hash"]
    else:
        fields = [item.get(key) for key in ("story_id", "story_title", "parent_id", "parent_text", "text", "fixture_label")]
        raw_hash = hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode("utf-8")).hexdigest()
    if item.get("supervision_hash"):
        return hashlib.sha256(f"{raw_hash}:{item['supervision_hash']}".encode("utf-8")).hexdigest()
    return raw_hash


def history(previous, train, holdout, roles):
    seen = dict(previous["seen"]) if previous else {}
    for item in train + holdout:
        seen[item["id"]] = {"fingerprint": fingerprint(item), "story_id": item["story_id"]}
    return {"roles": roles, "seen": seen}


def from_scratch(items, limit, validation_fraction):
    items = list(items)
    if len(items) < 4:
        raise ValueError("At least four usable comments are needed before training")
    train_indices, holdout_indices = split_threads(items[:limit], validation_fraction)
    train = [items[index] for index in train_indices]
    holdout = [items[index] for index in holdout_indices]
    roles = {item["story_id"]: "train" for item in train}
    roles.update({item["story_id"]: "holdout" for item in holdout})
    return {
        "train": train,
        "holdout": holdout,
        "history": history(None, train, holdout, roles),
        "counts": {"fresh_items": len(train), "fresh_holdout_items": len(holdout), "replay_items": 0},
    }


def incremental(items, previous, limit, config):
    roles = dict(previous["roles"])
    known = previous["seen"]
    replay_limit = max(0, int(config.get("incremental_replay_items", 32)))
    holdout_limit = max(1, int(config.get("incremental_holdout_items", 100)))
    candidates, replay, fresh_holdout, old_holdout = [], [], [], []
    scanned = set()
    for item in items:
        comment_id, story_id = item["id"], item["story_id"]
        if comment_id in scanned:
            continue
        scanned.add(comment_id)
        past = known.get(comment_id)
        if past and past["story_id"] != story_id:
            # A comment cannot move between groups after its text was used for training.
            continue
        is_fresh = past is None or past["fingerprint"] != fingerprint(item)
        if roles.get(story_id) == "holdout":
            sample = fresh_holdout if is_fresh else old_holdout
            if len(sample) < holdout_limit:
                sample.append(item)
        elif is_fresh:
            if len(candidates) < limit:
                candidates.append(item)
        elif len(replay) < replay_limit:
            replay.append(item)

    # Previously evaluated threads remain held out. When multiple new threads
    # arrive, reserve a fraction of those before fitting on any of them.
    new_stories = list(dict.fromkeys(item["story_id"] for item in candidates if item["story_id"] not in roles))
    validation_count = 0
    if len(new_stories) >= 2:
        fraction = float(config["validation_fraction"])
        validation_count = min(len(new_stories) - 1, max(1, round(len(new_stories) * fraction)))
    roles.update({
        story_id: "holdout" if index < validation_count else "train"
        for index, story_id in enumerate(new_stories)
    })
    fresh = []
    for item in candidates:
        if roles[item["story_id"]] == "holdout":
            fresh_holdout.append(item)
        else:
            fresh.append(item)

    old_reserve = min(len(old_holdout), holdout_limit // 2)
    fresh_holdout = fresh_holdout[:holdout_limit - old_reserve]
    holdout = fresh_holdout + old_holdout[:holdout_limit - len(fresh_holdout)]
    train = fresh + replay if fresh else []
    return {
        "train": train,
        "holdout": holdout,
        "history": history(previous, train, holdout, roles),
        "counts": {
            "fresh_items": len(fresh),
            "fresh_holdout_items": len(fresh_holdout),
            "replay_items": len(replay) if fresh else 0,
        },
    }
