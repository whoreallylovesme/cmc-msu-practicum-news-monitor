"""Fetch the historical HN training corpus and later, disjoint demo discussions."""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.config import load_config, load_monitoring
from lib.news import comments_as_dto

from sources import HackerNewsSource, aliases_for, clean_text, hacker_news_story, topic_patterns


HERE = Path(__file__).resolve().parent
START = datetime(2026, 7, 9, tzinfo=timezone.utc)
WINDOWS = 4
WINDOW_DAYS = 30
STORIES_PER_WINDOW = 3
COMMENTS_PER_STORY = 12
MAX_COMMENT_LENGTH = 1500
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TRAINING_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
TRAINING_END = datetime(2026, 1, 1, tzinfo=timezone.utc)
TRAINING_STORIES_PER_MONTH = 2
TRAINING_COMMENTS_PER_STORY = 40


def candidate(item):
    text = item.text
    parent = item.parent_text or ""
    return (
        25 <= len(text) <= MAX_COMMENT_LENGTH
        and len(parent) <= 1800
        and not EMAIL.search(text)
        and not EMAIL.search(parent)
    )


def sampling_order(item, query):
    return hashlib.sha256(f"{query}-{item.id}".encode()).digest()


def fetch(query, aliases=()):
    """Keep the original twelve-thread, 144-comment evaluation/demo sample."""
    cfg = load_config(HERE / "config.yaml")["live"]
    client = HackerNewsSource(cfg)
    end = START + timedelta(days=WINDOWS * WINDOW_DAYS)
    cutoff = int(end.timestamp()) - 1
    terms = aliases_for(query, aliases)
    patterns = topic_patterns(terms)
    selected = []
    for offset in range(WINDOWS):
        window_start = START + timedelta(days=offset * WINDOW_DAYS)
        window_end = window_start + timedelta(days=WINDOW_DAYS)
        window = []
        used_dates = set()
        for term in terms:
            search = client.get("/search", {
                "query": term,
                "tags": "story",
                "numericFilters": (
                    f"created_at_i>={int(window_start.timestamp())},"
                    f"created_at_i<{int(window_end.timestamp())},num_comments>=12"
                ),
                "hitsPerPage": 80,
            })
            for hit in search["hits"]:
                title = clean_text(hit.get("title"))
                if not any(pattern.search(title) for pattern in patterns):
                    continue
                date = hit["created_at"][:10]
                if date in used_dates:
                    continue
                story_id = str(hit["objectID"])
                story = client.get(f"/items/{story_id}")
                news = hacker_news_story(story, story_id, title, cutoff)
                comments = [item for item in news.comments if candidate(item)]
                if len(comments) < COMMENTS_PER_STORY:
                    continue
                sample = sorted(comments, key=lambda item: sampling_order(item, query))[:COMMENTS_PER_STORY]
                selected_news = replace(news, comments=tuple(sorted(sample, key=lambda item: item.date)))
                window.append({
                    "title": selected_news.title,
                    "date": selected_news.date.isoformat(),
                    "url": selected_news.url,
                    "comments": comments_as_dto([selected_news]),
                })
                used_dates.add(date)
                if len(window) == STORIES_PER_WINDOW:
                    break
            if len(window) == STORIES_PER_WINDOW:
                break
        if len(window) != STORIES_PER_WINDOW:
            raise RuntimeError(f"Only {len(window)} threads met the sampling rules in window {window_start.date()}")
        print(f"Selected {len(window)} threads from {window_start.date()} to {window_end.date()}", flush=True)
        selected.extend(window)
    return selected, end


def training_candidate(item):
    text = item.text
    parent = item.parent_text or ""
    return (
        30 <= len(text) <= 1200
        and len(parent) <= 1000
        and not EMAIL.search(text)
        and not EMAIL.search(parent)
    )


def training_order(item):
    return hashlib.sha256(f"quality-v1-{item.id}".encode()).digest()


def audit_order(item):
    return hashlib.sha256(f"final-blind-v1-{item.id}".encode()).digest()


def fetch_quality_audit(query, evaluation_stories, aliases=(), excluded_comment_ids=()):
    """Add five previously unused comments to each held-out discussion.

    Three explicit mentions plus two other replies improve target coverage. This
    stratified sample is not a population estimate and does not inspect sentiment.
    The original demo comments are always excluded before selecting these rows.
    """
    cfg = load_config(HERE / "config.yaml")["live"]
    client = HackerNewsSource(cfg)
    patterns = topic_patterns(aliases_for(query, aliases))
    excluded = set(excluded_comment_ids)
    stories = {}
    for story in evaluation_stories:
        for item in story["comments"]:
            excluded.add(item["id"])
            stories[item["story_id"]] = story
    cutoff = int((START + timedelta(days=WINDOWS * WINDOW_DAYS)).timestamp()) - 1
    selected = []
    for story_id in sorted(stories):
        raw = client.get(f"/items/{story_id}")
        news = hacker_news_story(raw, story_id, stories[story_id]["title"], cutoff)
        comments = sorted(
            (item for item in news.comments if item.id not in excluded and training_candidate(item)),
            key=audit_order,
        )
        direct = [item for item in comments if any(pattern.search(item.text) for pattern in patterns)]
        replies = [
            item for item in comments
            if item.parent_id != story_id and not any(pattern.search(item.text) for pattern in patterns)
        ]
        sample = {item.id: item for item in direct[:3] + replies[:2]}
        for item in comments:
            if len(sample) >= 5:
                break
            sample.setdefault(item.id, item)
        if len(sample) != 5:
            raise RuntimeError(
                f"Only {len(sample)} fresh audit comments met the rules in thread {story_id}; "
                "use --small-demo to fetch only the original sample"
            )
        chosen = replace(news, comments=tuple(sorted(sample.values(), key=lambda item: (item.date, item.id))))
        selected.append({
            "title": chosen.title,
            "date": chosen.date.isoformat(),
            "url": chosen.url,
            "comments": comments_as_dto([chosen]),
        })
        print(f"Selected {len(sample)} fresh audit comments in thread {story_id}", flush=True)
    return selected


def merge_stories(*parts):
    """Keep one story object while combining disjoint comment samples."""
    merged = {}
    seen_comments = set()
    for part in parts:
        for story in part:
            if not story["comments"]:
                continue
            story_id = story["comments"][0]["story_id"]
            target = merged.setdefault(story_id, {**story, "comments": []})
            for item in story["comments"]:
                if item["id"] in seen_comments:
                    raise ValueError(f"Comment {item['id']} occurs in more than one dataset part")
                if item["story_id"] != story_id:
                    raise ValueError("A story contains comments from different threads")
                seen_comments.add(item["id"])
                target["comments"].append(item)
    for story in merged.values():
        story["comments"].sort(key=lambda item: (item["published_at"], item["id"]))
    return list(merged.values())


def fetch_training(query, aliases=(), excluded_story_ids=()):
    """Fetch 48 earlier threads, keeping every later evaluation thread excluded.

    Sampling keeps direct target mentions, other root comments and other replies
    so unrelated discussion is represented rather than implicitly labeled relevant.
    The stable ID order is independent of any sentiment annotation.
    """
    cfg = load_config(HERE / "config.yaml")["live"]
    client = HackerNewsSource(cfg)
    terms = aliases_for(query, aliases)
    patterns = topic_patterns(terms)
    excluded = set(excluded_story_ids)
    selected = []
    selected_ids = set()
    text_hashes = set()
    month_start = TRAINING_START
    while month_start < TRAINING_END:
        month_end = (
            month_start.replace(year=month_start.year + 1, month=1)
            if month_start.month == 12 else month_start.replace(month=month_start.month + 1)
        )
        window = []
        for term in terms:
            search = client.get("/search", {
                "query": term,
                "tags": "story",
                "numericFilters": (
                    f"created_at_i>={int(month_start.timestamp())},"
                    f"created_at_i<{int(month_end.timestamp())},num_comments>=50"
                ),
                "hitsPerPage": 60,
            })
            for hit in search["hits"]:
                story_id = str(hit["objectID"])
                title = clean_text(hit.get("title"))
                if (
                    story_id in excluded or story_id in selected_ids
                    or not any(pattern.search(title) for pattern in patterns)
                ):
                    continue
                story = client.get(f"/items/{story_id}")
                news = hacker_news_story(story, story_id, title, int(month_end.timestamp()) - 1)
                comments = [
                    item for item in news.comments
                    if training_candidate(item) and hashlib.sha256(item.text.encode()).digest() not in text_hashes
                ]
                # A repeated quoted/reposted comment is one training example.
                unique_text = {}
                for item in sorted(comments, key=training_order):
                    unique_text.setdefault(hashlib.sha256(item.text.encode()).digest(), item)
                comments = list(unique_text.values())
                if len(comments) < TRAINING_COMMENTS_PER_STORY:
                    continue
                direct = [item for item in comments if any(pattern.search(item.text) for pattern in patterns)]
                direct_ids = {item.id for item in direct}
                root = [item for item in comments if item.parent_id == story_id and item.id not in direct_ids]
                replies = [item for item in comments if item.parent_id != story_id and item.id not in direct_ids]
                sample = {item.id: item for item in direct[:12] + root[:10] + replies[:18]}
                for item in comments:
                    if len(sample) >= TRAINING_COMMENTS_PER_STORY:
                        break
                    sample.setdefault(item.id, item)
                values = list(sample.values())
                text_hashes.update(hashlib.sha256(item.text.encode()).digest() for item in values)
                chosen = replace(news, comments=tuple(sorted(values, key=lambda item: (item.date, item.id))))
                window.append({
                    "title": chosen.title,
                    "date": chosen.date.isoformat(),
                    "url": chosen.url,
                    "comments": comments_as_dto([chosen]),
                })
                selected_ids.add(story_id)
                if len(window) == TRAINING_STORIES_PER_MONTH:
                    break
            if len(window) == TRAINING_STORIES_PER_MONTH:
                break
        if len(window) != TRAINING_STORIES_PER_MONTH:
            raise RuntimeError(
                f"Only {len(window)} training threads met the sampling rules in {month_start:%Y-%m}; "
                "choose a query with enough history or use --small-demo"
            )
        selected.extend(window)
        print(f"Selected {len(window)} training threads in {month_start:%Y-%m}", flush=True)
        month_start = month_end
    return selected


def dataset_part(stories, start, end):
    items = [item for story in stories for item in story["comments"]]
    return {
        "story_window_start_inclusive": start.isoformat(),
        "story_window_end_exclusive": end.isoformat(),
        "thread_count": len({item["story_id"] for item in items}),
        "comment_count": len(items),
        "story_ids": sorted({item["story_id"] for item in items}),
        "comment_ids": sorted(item["id"] for item in items),
    }


def main():
    monitoring = load_monitoring()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local-data/hn_snapshot.raw.json")
    parser.add_argument("--query", default=monitoring["query"], help="Override monitoring.query for this snapshot")
    parser.add_argument("--small-demo", action="store_true", help="Fetch only the original later 144-comment demo")
    args = parser.parse_args()
    query = args.query.strip()
    if not query:
        parser.error("query must not be blank")
    aliases = monitoring["aliases"] if query == monitoring["query"] else []
    original_evaluation, end = fetch(query, aliases)
    audit = [] if args.small_demo else fetch_quality_audit(query, original_evaluation, aliases)
    evaluation = merge_stories(original_evaluation, audit)
    evaluation_ids = {item["story_id"] for story in evaluation for item in story["comments"]}
    training = [] if args.small_demo else fetch_training(query, aliases, evaluation_ids)
    stories = training + evaluation
    items = [item for story in stories for item in story["comments"]]
    data = json.dumps(stories, ensure_ascii=False, indent=2) + "\n"
    manifest = {
        "title": f"Hacker News comments mentioning {query} through their discussion context",
        "dataset_kind": "public_discussion_snapshot",
        "attribution": "Public Hacker News discussions and comments; individual originals linked by URL.",
        "source_api": "https://hn.algolia.com/api/v1",
        "site": "https://news.ycombinator.com/",
        "query": query,
        "aliases": aliases,
        "story_window_start_inclusive": (TRAINING_START if training else START).isoformat(),
        "story_window_end_exclusive": end.isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "thread_count": len(stories),
        "comment_count": len(items),
        "min_comment_date": min(item["published_at"] for item in items),
        "max_comment_date": max(item["published_at"] for item in items),
        "parts": {
            "training": dataset_part(training, TRAINING_START, TRAINING_END),
            "evaluation": dataset_part(evaluation, START, end),
            "audit": dataset_part(audit, START, end),
        },
        "sampling": (
            "Evaluation/demo: three title-matching discussions on different submission dates per 30-day window, "
            "twelve comments per discussion sampled by SHA256(query-ID). "
            "Unless --small-demo, add five previously unused audit comments per evaluation thread by "
            "SHA256(final-blind-v1-ID): up to three direct mentions and two other replies, then fill in hash order. "
            "Audit comment length30..1200, parent<=1000; these sixty rows remain part of the held-out threads "
            "and are also listed separately in parts.audit. Their target-coverage sampling is not a population estimate. "
            "Training, unless --small-demo: two title-matching discussions per calendar month in 2024-2025, "
            "forty comments each by SHA256(quality-v1-ID), selecting up to twelve direct mentions, ten other "
            "root comments and eighteen other replies before filling in hash order. Duplicate training texts "
            "and all evaluation threads are excluded. Training comment length30..1200, parent<=1000; "
            "evaluation comment length25..1500, parent<=1800. All text is HTML-normalized; detected email "
            "addresses are skipped. No author fields or sentiment labels are retained."
        ),
        "limitation": "This is a purposive sample of one entity, not a representative census of Hacker News. Index edits can change regenerated results. No separate license for the underlying comments is asserted.",
        "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(data, encoding="utf-8")
    metadata = args.output.with_suffix(".metadata.json")
    metadata.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(items)} comments from {len(stories)} threads to {args.output}")


if __name__ == "__main__":
    main()
