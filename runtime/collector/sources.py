"""Source adapters and shared continuation cursor for the Collector."""

import base64
import binascii
import hashlib
import json
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from itertools import groupby
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from lib.news import News, NewsComment, NewsProvider


class InvalidCursor(ValueError):
    """The cursor is invalid for the requested collection scope."""


class SourceUnavailable(RuntimeError):
    """Hacker News could not return a usable response."""


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "div", "blockquote", "pre", "li"}:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def clean_text(value):
    if not isinstance(value, str):
        return ""
    parser = TextExtractor()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


def aliases_for(entity, aliases):
    unique = {}
    for term in [entity, *aliases]:
        unique.setdefault(term.casefold(), term)
    return list(unique.values())


def topic_patterns(terms):
    return [re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE) for term in terms]


def cursor_scope(source, terms):
    payload = json.dumps([source, terms], ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def encode_cursor(scope, state):
    payload = json.dumps({"scope": scope, **state}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_cursor(cursor, scope, keys):
    try:
        padding = "=" * (-len(cursor) % 4)
        value = json.loads(base64.b64decode(cursor + padding, altchars=b"-_", validate=True))
    except (ValueError, TypeError, UnicodeError, binascii.Error) as exc:
        raise InvalidCursor("Malformed continuation cursor") from exc
    if not isinstance(value, dict) or value.pop("scope", None) != scope:
        raise InvalidCursor("Cursor belongs to another source or topic")
    if set(value) != set(keys) or any(type(value[key]) is not int or value[key] < 0 for key in keys):
        raise InvalidCursor("Malformed cursor state")
    return value


def read_date(value):
    if not isinstance(value, str):
        raise ValueError("Provider dates must be ISO strings")
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("Provider dates must include a time zone")
    return date.astimezone(timezone.utc)


def archive_news(story):
    raw_comments = story["comments"]
    first = raw_comments[0]
    story_id = first["story_id"]
    comments = tuple(
        NewsComment(
            id=item["id"], date=read_date(item["published_at"]), text=item["text"], url=item["url"],
            parent_id=item.get("parent_id"), parent_text=item.get("parent_text"),
            fixture_label=item.get("fixture_label"),
        )
        for item in raw_comments
    )
    date = read_date(story["date"]) if "date" in story else min(comment.date for comment in comments)
    default_url = (
        f"https://news.ycombinator.com/item?id={story_id}" if story_id.isdecimal() else first["url"]
    )
    return News(
        id=story_id, source=first["source"], date=date, title=story["title"],
        description=story.get("description", ""), url=story.get("url", default_url), comments=comments,
    )


class ArchiveSource(NewsProvider):
    def __init__(self, path, name="archive", description="Saved discussions", chronological=False):
        super().__init__(name, description)
        with path.open(encoding="utf-8") as fixture:
            self.stories = [archive_news(story) for story in json.load(fixture) if story["comments"]]
        self.chronological = chronological

    def batch(self, terms, cursor, limit):
        scope = cursor_scope(self.name, terms)
        offset = decode_cursor(cursor, scope, ["offset"])["offset"] if cursor else 0
        patterns = topic_patterns(terms)
        items = [
            (story, comment)
            for story in self.stories
            if any(pattern.search(story.title) for pattern in patterns)
            for comment in story.comments
        ]
        if self.chronological:
            items.sort(key=lambda pair: (pair[1].date, pair[1].id))
        page = items[offset:offset + limit]
        end = offset + len(page)
        next_cursor = encode_cursor(scope, {"offset": end}) if end < len(items) else None
        news = []
        for _, group in groupby(page, key=lambda pair: pair[0].id):
            story, first_comment = next(group)
            comments = (first_comment, *(comment for _, comment in group))
            news.append(replace(story, comments=comments))
        return news, next_cursor


class HackerNewsSource(NewsProvider):
    requires_network = True

    def __init__(self, config):
        super().__init__("hacker_news", "Live Hacker News discussions via Algolia HN Search")
        self.base_url = config["base_url"].rstrip("/")
        self.timeout = config["timeout_seconds"]
        self.interval = config["min_request_interval_seconds"]
        self.lookback = config["lookback_days"] * 86400
        self.page_size = config["stories_per_page"]
        self.request_budget = config["requests_per_batch"]
        self.max_response_bytes = config["max_response_bytes"]
        self.lock = Lock()
        self.next_request_at = 0.0

    def get(self, path, params=None):
        query = "?" + urlencode(params) if params else ""
        request = Request(self.base_url + path + query, headers={
            "Accept": "application/json",
            "User-Agent": "cmc-msu-practicum-news-monitor/1.0",
        })
        with self.lock:
            delay = self.next_request_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_request_at = time.monotonic() + self.interval
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read(self.max_response_bytes + 1)
            if len(body) > self.max_response_bytes:
                raise SourceUnavailable("Hacker News response exceeded the size limit")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise SourceUnavailable("Hacker News returned an invalid response")
            return data
        except HTTPError as exc:
            if exc.code == 404 and path.startswith("/items/"):
                return {}
            raise SourceUnavailable(f"Hacker News returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise SourceUnavailable("Hacker News API request failed") from exc

    def batch(self, terms, cursor, limit):
        scope = cursor_scope(self.name, terms)
        keys = ["until", "alias", "page", "hit", "comment"]
        state = decode_cursor(cursor, scope, keys) if cursor else dict(
            until=int(time.time()), alias=0, page=0, hit=0, comment=0,
        )
        if state["until"] > int(time.time()) + 1 or state["alias"] > len(terms):
            raise InvalidCursor("Invalid Hacker News cursor position")
        patterns = topic_patterns(terms)
        items = []
        count = 0
        requests_left = self.request_budget
        while state["alias"] < len(terms) and count < limit and requests_left > 1:
            alias = state["alias"]
            search = self.get("/search_by_date", {
                "query": terms[alias],
                "tags": "story",
                "numericFilters": f'created_at_i>={state["until"] - self.lookback},created_at_i<={state["until"]}',
                "page": state["page"],
                "hitsPerPage": self.page_size,
            })
            requests_left -= 1
            hits = search.get("hits")
            pages = search.get("nbPages")
            if not isinstance(hits, list) or type(pages) is not int:
                raise SourceUnavailable("Hacker News search returned an invalid page")
            while state["hit"] < len(hits) and count < limit and requests_left > 0:
                hit = hits[state["hit"]]
                if not isinstance(hit, dict):
                    raise SourceUnavailable("Hacker News search returned an invalid story")
                title = clean_text(hit.get("title"))
                first_alias = next((i for i, pattern in enumerate(patterns) if pattern.search(title)), None)
                story_id = str(hit.get("objectID", ""))
                if story_id.isdecimal() and first_alias == alias:
                    story = self.get(f"/items/{story_id}")
                    requests_left -= 1
                    news = hacker_news_story(story, story_id, title, state["until"])
                    offset = state["comment"]
                    page = news.comments[offset:offset + limit - count]
                    if page:
                        items.append(replace(news, comments=page))
                        count += len(page)
                    if offset + len(page) < len(news.comments):
                        state["comment"] += len(page)
                        break
                state["comment"] = 0
                state["hit"] += 1
            if state["hit"] >= len(hits):
                state["page"] += 1
                state["hit"] = 0
                state["comment"] = 0
                if state["page"] >= pages or not hits:
                    state.update(alias=alias + 1, page=0)
        next_cursor = encode_cursor(scope, state) if state["alias"] < len(terms) else None
        return items, next_cursor


def timestamp(node):
    stamp = node.get("created_at_i")
    if type(stamp) is int:
        return stamp
    try:
        return int(read_date(node.get("created_at") or "").timestamp())
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def hacker_news_story(story, story_id, title, cutoff):
    comments = tuple(thread_comments(story, story_id, cutoff))
    stamp = timestamp(story)
    if stamp is not None and 0 <= stamp <= cutoff:
        date = datetime.fromtimestamp(stamp, tz=timezone.utc)
    else:
        date = min((comment.date for comment in comments), default=datetime.fromtimestamp(0, tz=timezone.utc))
    return News(
        id=story_id, source="hacker_news", date=date, title=title, description=clean_text(story.get("text")),
        url=f"https://news.ycombinator.com/item?id={story_id}", comments=comments,
    )


def thread_comments(story, story_id, cutoff):
    """Flatten nested replies, preserving each immediate parent and a fixed time window."""
    root_children = story.get("children") or []
    if not isinstance(root_children, list):
        raise SourceUnavailable("Hacker News thread returned invalid replies")
    pending = [(child, story_id, None) for child in reversed(root_children)]
    items = []
    seen = set()
    while pending:
        node, parent_id, parent_text = pending.pop()
        if not isinstance(node, dict):
            raise SourceUnavailable("Hacker News thread returned an invalid comment")
        comment_id = str(node.get("id", ""))
        if not comment_id.isdecimal() or comment_id in seen:
            continue
        seen.add(comment_id)
        text = clean_text(node.get("text"))
        stamp = timestamp(node)
        if text and stamp is not None and 0 <= stamp <= cutoff:
            items.append(NewsComment(
                id=comment_id, date=datetime.fromtimestamp(stamp, tz=timezone.utc), text=text,
                url=f"https://news.ycombinator.com/item?id={comment_id}",
                parent_id=parent_id, parent_text=parent_text,
            ))
        children = node.get("children") or []
        if not isinstance(children, list):
            raise SourceUnavailable("Hacker News comment returned invalid replies")
        pending.extend((child, comment_id, text or None) for child in reversed(children))
    return sorted(items, key=lambda item: (item.date, int(item.id)), reverse=True)
