"""Canonical article and comment types shared by Collector providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class NewsComment:
    id: str
    date: datetime
    text: str
    url: str
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None


@dataclass(frozen=True)
class News:
    id: str
    source: str
    date: datetime
    title: str
    description: str
    url: str
    comments: Tuple[NewsComment, ...] = ()


class NewsProvider(ABC):
    """Retrieve one bounded page; the limit counts comments, not articles."""

    requires_network = False

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def batch(
        self, terms: List[str], cursor: Optional[str], limit: int,
    ) -> Tuple[List[News], Optional[str]]:
        """Return articles with their page of comments and an opaque continuation."""


def comments_as_dto(news_items: Iterable[News]) -> list[dict]:
    """Flatten normalized provider records to the public Collector wire format."""
    return [
        {
            "id": comment.id,
            "story_id": story.id,
            "parent_id": comment.parent_id,
            "published_at": comment.date.isoformat(),
            "story_title": story.title,
            "text": comment.text,
            "parent_text": comment.parent_text,
            "url": comment.url,
            "source": story.source,
        }
        for story in news_items
        for comment in story.comments
    ]
