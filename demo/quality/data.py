"""Скачать закрепленные человеческие benchmarks в локальный кеш."""

import csv
import hashlib
import io
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
import re
import statistics
import urllib.request


LOGGER = logging.getLogger(__name__)
NEWS_REVISION = "6b838e00f54423c253806327a0ae24dbffa24c9e"
NEWS_BASE = (f"https://raw.githubusercontent.com/fhamborg/NewsMTSC/{NEWS_REVISION}/"
             "NewsSentiment/experiments/default/datasets/")
NEWS_DIGESTS = {
    ("rw", "train"): "e78c7964aad1d5b7dbc66cb3a05d5885dbf2e29a2a3171f0d75b497bf18f3621",
    ("rw", "dev"): "78b5dde35a97b452035c07c142fac5ecf44dd998217921cff62a1519ed01e952",
    ("rw", "test"): "9bb6bf992bda28dda7eb0508a55964dbdd1b4a4a0dcc66f5c5417943a854eb2b",
    ("mt", "train"): "e78c7964aad1d5b7dbc66cb3a05d5885dbf2e29a2a3171f0d75b497bf18f3621",
    ("mt", "dev"): "9dc6480071836094894ee9606cd6bd810d3edeb2d37638d751e55bbb7d375aaa",
    ("mt", "test"): "5019b47c190a32ee88687bd383cfe1c8e921e497af5990abac1f5d1c0479a993",
}
POLARITY = {-1: "negative", 0: "neutral", 1: "positive"}
WEATHER_REVISION = "f958365e4d8aa13a926a7e3a4505cff13f3bbeed"
WEATHER_URL = (f"https://raw.githubusercontent.com/snorkel-team/snorkel/{WEATHER_REVISION}/"
               "tutorials/crowdsourcing/data/weather-evaluated-agg-DFE.csv")
WEATHER_SHA256 = "3d89efbdb70b5c6712ad66595e32209b9b69ccea373f3d57343908ea6c45752c"
AIRLINE_REVISION = "dec17e9391b767791e3808a655654467605a9d49"
AIRLINE_BASE = f"https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment/resolve/{AIRLINE_REVISION}/"
AIRLINE_SHA256 = "ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e"
AIRLINE_CARD_SHA256 = "568b2288639e85b3e36a504f70d37098e2924df188f7d4b7b9466019d753edab"
AIRLINES = {"American": "American Airlines", "United": "United Airlines", "Southwest": "Southwest Airlines",
            "US Airways": "US Airways", "Virgin America": "Virgin America"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(directory, url, filename, expected, maximum_bytes=10_000_000):
    raw_dir = Path(directory) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / filename
    if path.exists():
        raw = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "HNMonitorQuality/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(maximum_bytes + 1)
        assert len(raw) <= maximum_bytes, f"Unexpected file size: {url}"
        assert digest(raw) == expected, f"Changed pinned source: {url}"
        path.write_bytes(raw)
    assert digest(raw) == expected, f"Changed cached source: {path}"
    return raw


def text_key(text):
    return digest(" ".join(text.casefold().split()).encode())


def news_group(row):
    # POLUSA keeps all mentions/sentences from one source ID together. In
    # AllSides IDs, multiple first-number documents share the second numeric
    # block; conservatively keep that larger block together, too.
    match = re.match(r"(polusa_v\d+_\d+)_", row["id"])
    if match:
        return match.group(1)
    match = re.match(r"allsides_\d+_(\d+)_", row["id"])
    assert match, f"Unknown source ID schema: {row['id']}"
    return "allsides_source_block:" + match.group(1)


def canonical_news(row, subset, split):
    assert row["polarity"] in POLARITY
    assert row["sentence"][row["from"]:row["to"]] == row["mention"]
    return {
        "id": row["id"], "entity": row["mention"], "text": row["sentence"],
        "story_title": "", "parent_text": "", "gold_polarity": POLARITY[row["polarity"]],
        "gold_relevant": True, "split": split, "dataset": "newsmtsc_" + subset,
        "group_id": news_group(row), "text_sha256": text_key(row["sentence"]),
        "gold_intensity": None, "target_start": row["from"], "target_end": row["to"],
    }


def write_jsonl(path, rows):
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(payload)
    return {"path": path.name, "rows": len(rows), "sha256": digest(payload),
            "polarity_counts": dict(Counter(row["gold_polarity"] for row in rows)),
            "relevance_counts": dict(Counter(str(row["gold_relevant"]) for row in rows))}


def prepare_news(directory):
    data, sources = {}, []
    for (subset, split), expected in NEWS_DIGESTS.items():
        filename = f"newsmtsc-{subset}-{split}.jsonl"
        url = NEWS_BASE + f"newsmtsc-{subset}-hf/{split}.jsonl"
        raw = fetch(directory, url, filename, expected)
        original = [json.loads(line) for line in raw.splitlines()]
        data[(subset, split)] = [canonical_news(row, subset, split) for row in original]
        sources.append({"url": url, "revision": NEWS_REVISION, "sha256": expected,
                        "raw_rows": len(original), "bytes": len(raw)})

    # Test sets stay verbatim. Train/selection data cannot share articles or sentences
    # with either test set. RW and MT share their training set by official design.
    test_groups = {row["group_id"] for key, rows in data.items() if key[1] == "test" for row in rows}
    test_texts = {row["text_sha256"] for key, rows in data.items() if key[1] == "test" for row in rows}
    removed = {}
    for subset in ("rw", "mt"):
        dev = data[(subset, "dev")]
        data[(subset, "dev")] = [row for row in dev if row["group_id"] not in test_groups
                                  and row["text_sha256"] not in test_texts]
        removed[f"{subset}_dev_test_overlap"] = len(dev) - len(data[(subset, "dev")])
    dev_groups = {row["group_id"] for key, rows in data.items() if key[1] == "dev" for row in rows}
    dev_texts = {row["text_sha256"] for key, rows in data.items() if key[1] == "dev" for row in rows}
    forbidden_groups = test_groups | dev_groups
    forbidden_texts = test_texts | dev_texts
    for subset in ("rw", "mt"):
        train = data[(subset, "train")]
        data[(subset, "train")] = [row for row in train if row["group_id"] not in forbidden_groups
                                    and row["text_sha256"] not in forbidden_texts]
        removed[f"{subset}_train_eval_overlap"] = len(train) - len(data[(subset, "train")])
    files = {}
    for (subset, split), rows in data.items():
        assert rows, f"No data after leakage exclusion for {subset}/{split}"
        assert len({row["id"] for row in rows}) == len(rows)
        files[f"newsmtsc_{subset}_{split}"] = write_jsonl(Path(directory) / f"newsmtsc_{subset}_{split}.jsonl", rows)
    return {
        "name": "NewsMTSC", "license": "MIT (author's HF card and repository)",
        "license_url": "https://huggingface.co/datasets/fhamborg/news_sentiment_newsmtsc",
        "paper": "https://aclanthology.org/2021.eacl-main.142/", "sources": sources,
        "labels": "Five human coders; target polarity -1/0/+1. Annotated mention is relevant by task definition.",
        "limitation": "No irrelevant examples, no five-level intensity, no conversation context; political-news domain.",
        "split_policy": "Original test identities/content/labels unchanged. Remove train/dev source-ID groups or normalized-sentence collisions with all test; remove train collisions with retained dev. POLUSA uses its source ID; AllSides conservatively groups by the shared second numeric ID block. RW/MT test may overlap each other and must not be pooled as independent samples.",
        "removed": removed, "files": files,
    }


def prepare_weather(directory):
    raw = fetch(directory, WEATHER_URL, "weather-evaluated-agg-DFE.csv", WEATHER_SHA256)
    source_rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    mapping = {"Negative": "negative", "Neutral / author is just sharing information": "neutral",
               "Positive": "positive", "Tweet not related to weather condition": None}
    rows, omitted = [], Counter()
    for row in source_rows:
        # Do not use labels rejected or left undecided by the independent evaluators.
        if row["correct_category"] != "Yes":
            omitted["independent_evaluation_" + row["correct_category"]] += 1
            continue
        if row["sentiment"] not in mapping:
            omitted["original_label_uncertain"] += 1
            continue
        value = mapping[row["sentiment"]]
        rows.append({
            "id": "weather:" + row["tweet_id"], "entity": "weather conditions", "text": row["tweet_body"],
            "story_title": "", "parent_text": "", "gold_polarity": value,
            "gold_relevant": value is not None, "split": None, "dataset": "weather_sentiment",
            "group_id": "tweet:" + row["tweet_id"], "text_sha256": text_key(row["tweet_body"]),
            "gold_intensity": None, "annotation_confirmation_confidence": float(row["correct_category_conf"]),
        })
    assert len({row["id"] for row in rows}) == len(rows)
    # No official split: reserve all weather rows for independent evaluation,
    # using a reproducible 20% development / 80% final test split per human class.
    data = {"dev": [], "test": []}
    for relevant, polarity in ((True, "negative"), (True, "neutral"), (True, "positive"), (False, None)):
        selected = [row for row in rows if row["gold_relevant"] == relevant and row["gold_polarity"] == polarity]
        selected.sort(key=lambda row: digest(("weather-split-v1:" + row["text_sha256"]).encode()))
        count = max(1, round(len(selected) * 0.2))
        for index, row in enumerate(selected):
            split = "dev" if index < count else "test"
            data[split].append({**row, "split": split})
    # Keep repeated texts as a group. If a duplicate straddles splits, remove
    # only the dev copy. Test identities/content/labels remain untouched.
    test_texts = {row["text_sha256"] for row in data["test"]}
    before = len(data["dev"])
    data["dev"] = [row for row in data["dev"] if row["text_sha256"] not in test_texts]
    omitted["dev_exact_text_overlap_test"] += before - len(data["dev"])
    assert not {row["text_sha256"] for row in data["dev"]} & test_texts
    files = {"weather_sentiment_" + split: write_jsonl(Path(directory) / f"weather_sentiment_{split}.jsonl", items)
             for split, items in data.items()}
    return {
        "name": "CrowdFlower Weather Sentiment (independently evaluated)",
        "license": "Original dataset license not verified from surviving primary source; local evaluation only, not redistributed. Mirror repository is Apache-2.0; this is not an assertion that underlying tweets are Apache-2.0.",
        "license_url": f"https://github.com/snorkel-team/snorkel/blob/{WEATHER_REVISION}/LICENSE",
        "original_source": "https://www.crowdflower.com/data-for-everyone/",
        "research_mirror": f"https://github.com/snorkel-team/snorkel/tree/{WEATHER_REVISION}/tutorials/crowdsourcing",
        "sources": [{"url": WEATHER_URL, "revision": WEATHER_REVISION, "sha256": WEATHER_SHA256,
                     "bytes": len(raw), "raw_rows": len(source_rows)}],
        "labels": "Original20crowd judgments/tweet; separate independent correctness evaluation. Retain only correct_category=Yes and definite sentiment/relevance labels, with no model labels or synthetic target swaps.",
        "limitation": "One topic, old short tweets, no conversation context or five-level intensity. Relevant factual weather tweets are neutral, not F.",
        "split_policy": "No official splits; deterministic human-class-stratified20%dev/80%test. Weather has no training split and is never mixed into student training. Avoid selecting repeatedly on test.",
        "omitted": dict(omitted), "files": files,
    }



def prepare_airline(directory):
    """Человеческая тональность к авиакомпании без размеченных span и train."""
    raw = fetch(directory, AIRLINE_BASE + "Tweets.csv", "airline-Tweets.csv", AIRLINE_SHA256)
    fetch(directory, AIRLINE_BASE + "README.md", "airline-README.md", AIRLINE_CARD_SHA256)
    original = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    rows, seen, excluded, duplicates = [], set(), Counter(), 0
    for index, row in enumerate(original):
        if row["airline"] not in AIRLINES:
            excluded[row["airline"]] += 1
            continue
        assert row["airline_sentiment"] in {"negative", "neutral", "positive"} and row["text"]
        key = (row["tweet_id"], row["airline"], row["text"], row["airline_sentiment"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        target = AIRLINES[row["airline"]]
        rows.append({
            "id": "airline:" + row["tweet_id"] + ":" + str(index), "source_id": row["tweet_id"],
            "entity": target, "aliases": [row["airline"]] if row["airline"] != target else [],
            "text": row["text"], "story_title": "", "parent_text": "", "gold_polarity": row["airline_sentiment"],
            "gold_relevant": True, "gold_intensity": None, "split": None, "dataset": "airline_sentiment",
            "group_id": None, "text_sha256": text_key(row["text"]), "source_airline": row["airline"],
            "annotation_confidence": float(row["airline_sentiment_confidence"]), "_author": row["name"].strip().casefold(),
        })
    # One connected component contains shared authors, copied text and tweet IDs.
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    indices = {}
    for index, row in enumerate(rows):
        keys = [("text", row["text_sha256"]), ("tweet", row["source_id"])]
        if row["_author"]:
            keys.append(("author", row["_author"]))
        for key in keys:
            if key in indices:
                first, second = find(index), find(indices[key])
                parents[max(first, second)] = min(first, second)
            else:
                indices[key] = index
    components = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    partitions = {"dev": [], "test": []}
    for members in components.values():
        group = "airline-author-text:" + digest("\n".join(sorted(rows[index]["id"] for index in members)).encode())
        split = "dev" if int(digest(("airline-transfer-v1:" + group).encode())[:8], 16) % 5 == 0 else "test"
        for index in members:
            rows[index]["group_id"] = group
            partitions[split].append(rows[index])
    selected, files = {}, {}

    def confidence(items):
        values = [row["annotation_confidence"] for row in items]
        return {"minimum": min(values), "median": statistics.median(values), "mean": statistics.mean(values),
                "maximum": max(values), "below_0_5": sum(value < .5 for value in values), "exactly_1": sum(value == 1 for value in values)}

    for split, quotas in (("dev", (100, 100, 100)), ("test", (334, 333, 333))):
        sample = []
        for label, count in zip(("negative", "neutral", "positive"), quotas):
            pool = sorted((row for row in partitions[split] if row["gold_polarity"] == label),
                          key=lambda row: digest(("airline-transfer-sample-v1:" + row["id"]).encode()))
            assert len(pool) >= count
            sample.extend(pool[:count])
        sample.sort(key=lambda row: digest(("airline-transfer-order-v1:" + row["id"]).encode()))
        selected[split] = sample
        canonical = [{key: split if key == "split" else value for key, value in row.items() if key != "_author"} for row in sample]
        name = "airline_sentiment_" + split
        files[name] = write_jsonl(Path(directory) / (name + ".jsonl"), canonical)
        files[name].update(targets=dict(Counter(row["entity"] for row in sample)),
                           groups=len({row["group_id"] for row in sample}), confidence=confidence(sample))
    for key in ("_author", "text_sha256", "source_id", "group_id"):
        assert not {row[key] for row in selected["dev"]} & {row[key] for row in selected["test"]}, key
    by_id = defaultdict(set)
    for row in rows:
        by_id[row["source_id"]].add(row["gold_polarity"])
    conflicts = {key for key, values in by_id.items() if len(values) > 1}
    return {
        "name": "Twitter US Airline Sentiment", "dataset": "airline_sentiment",
        "source": {"publisher": "CrowdFlower / Kaggle", "revision": AIRLINE_REVISION,
                   "url": AIRLINE_BASE + "Tweets.csv", "sha256": AIRLINE_SHA256, "raw_rows": len(original),
                   "card_url": AIRLINE_BASE + "README.md", "card_sha256": AIRLINE_CARD_SHA256},
        "license": "CC-BY-NC-SA-4.0 according to the pinned HF dataset card; local evaluation only, not redistributed",
        "license_url": f"https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment/blob/{AIRLINE_REVISION}/README.md",
        "curation": {"excluded_source_target_buckets": dict(excluded),
                     "reason": "Entire Delta bucket excluded: 2218 of2222 tweets address @JetBlue; no guessed per-row target repair",
                     "exact_duplicate_annotations_removed": duplicates, "target_mapping": AIRLINES,
                     "conflicting_source_tweet_sentiments_retained": len(conflicts), "confidence_filter": None,
                     "conflict_rows_in_dev": sum(row["source_id"] in conflicts for row in selected["dev"]),
                     "conflict_rows_in_test": sum(row["source_id"] in conflicts for row in selected["test"]),
                     "eligible_rows": len(rows), "eligible_groups": len(components)},
        "split_policy": "Hash connected components of authors, normalized text and tweet IDs into dev/test pools; fixed balanced quotas300dev/1000test; all remaining rows unused, no training split",
        "labels": "Original human crowd sentiment towards the source airline; supplied brand name, raw text unchanged, no target-span markers",
        "limitation": "All examples relevant; no F or five-level intensity gold. Old tweets may occur in pretraining; balanced diagnostic is not population sentiment",
        "files": files,
    }


def canonical_adaptation(row, selected, split, entity):
    """Воспроизвести человеческую разметку; unknown не превращается в C/F."""
    from lib.training_sources import _weather

    result = _weather(row, {"entity": entity})
    votes = [float(row[f"s{i}"]) for i in range(1, 6)]
    winner = max(range(5), key=votes.__getitem__)
    assert result is not None and winner != 0 and votes.count(votes[winner]) == 1
    target, text, probabilities, weight = result
    return {
        "id": selected["id"], "group_id": selected["group_id"], "entity": target, "text": text,
        "coarse_probabilities": probabilities, "sample_weight": weight, "split": split,
        "dataset": "weather_adaptation", "story_title": "", "parent_text": "",
        "gold_relevant": winner != 4,
        "gold_polarity": None if winner == 4 else ("negative", "neutral", "positive")[winner - 1],
        "gold_intensity": None, "source_known_mass": weight,
    }


def prepare_weather_adaptation(directory, selector_path):
    """Только локальная реконструкция закреплённых DEV/FINAL, без новых разбиений."""
    import yaml

    descriptor = yaml.safe_load(Path(selector_path).read_text())
    assert descriptor["version"] == 1 and descriptor["dataset"] == "weather_adaptation"
    source = descriptor["source"]
    raw = fetch(directory, source["url"], "partly-sunny-original-train.csv", source["sha256"], 64 * 1024 ** 2)
    assert len(raw) == source["size_bytes"]
    original = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        identity = "partly-sunny:" + row["id"]
        assert identity not in original
        original[identity] = row
    files, used_ids, used_groups = {}, set(), set()
    for split in ("dev", "test"):
        spec = descriptor["splits"][split]
        selected = spec["items"]
        ids, groups = {row["id"] for row in selected}, {row["group_id"] for row in selected}
        assert len(selected) == len(ids) == len(groups) and not ids & used_ids and not groups & used_groups
        rows = [canonical_adaptation(original[row["id"]], row, split, descriptor["entity"]) for row in selected]
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                          + "\n" for row in rows).encode()
        assert digest(payload) == spec["canonical_sha256"], "Changed frozen Weather benchmark"
        filename = f"weather_adaptation_{split}.jsonl"
        (Path(directory) / filename).write_bytes(payload)
        files[f"weather_adaptation_{split}"] = {"path": filename, "rows": len(rows), "sha256": digest(payload)}
        used_ids.update(ids)
        used_groups.update(groups)
    return {"name": "Partly Sunny — fixed independent human Weather benchmark", "dataset": "weather_adaptation",
            "source": source, "license": source["license"], "reference": descriptor["reference"],
            "split_policy": descriptor["split_policy"], "files": files}


def prepare(directory, weather_benchmark=None):
    """Воспроизводимые данные и manifest; ни одна разметка не создается моделью."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "purpose": "Independent human evaluation; never teacher-agreement accuracy.",
                "datasets": [prepare_news(directory), prepare_weather(directory), prepare_airline(directory)]}
    if weather_benchmark is not None:
        manifest["datasets"].append(prepare_weather_adaptation(directory, weather_benchmark))
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for dataset in manifest["datasets"]:
        LOGGER.info("%s: %s", dataset["name"], {key: row["rows"] for key, row in dataset["files"].items()})
    return manifest


def read_split(directory, dataset, split):
    """Проверить canonical-файл по manifest перед подсчетом метрик."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    key = f"{dataset}_{split}"
    descriptors = {name: item for source in manifest["datasets"] for name, item in source["files"].items()}
    descriptor = descriptors[key]
    raw = (directory / descriptor["path"]).read_bytes()
    assert digest(raw) == descriptor["sha256"], "Benchmark изменен; выполните operation: prepare"
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == descriptor["rows"]
    assert all(row["dataset"] == dataset and row["split"] == split for row in rows)
    return rows
