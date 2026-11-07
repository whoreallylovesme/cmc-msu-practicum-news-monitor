"""REST teacher DTOs and full-context cache identity shared with its client."""

import hashlib
import json
from collections.abc import Mapping
from typing import Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from lib.mentions import MentionMatcher
from lib.sentiment import CHOICES


INPUT_HASH_VERSION = "teacher-full-context-v1"


class MatchingPolicy(BaseModel):
    allow_typos: bool = True
    allow_context: bool = True
    context_hints: list[str] = Field(default_factory=list, max_length=32)


class TeacherComment(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    story_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=100000)
    story_title: str = Field(default="", max_length=10000,
                             validation_alias=AliasChoices("story_title", "title"))
    parent_text: Optional[str] = Field(default=None, max_length=100000,
                                       validation_alias=AliasChoices("parent_text", "parent"))
    parent_id: Optional[str] = Field(default=None, max_length=128)


class TeacherRequest(BaseModel):
    entity: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=32)
    matching: MatchingPolicy = Field(default_factory=MatchingPolicy)
    items: list[TeacherComment] = Field(min_length=1, max_length=100)

    @field_validator("entity")
    @classmethod
    def valid_entity(cls, value):
        value = value.strip()
        MentionMatcher(value)
        return value

    @model_validator(mode="after")
    def unique_ids(self):
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Teacher request comment IDs must be unique")
        return self


class TeacherLabel(BaseModel):
    comment_id: str
    probabilities: list[float] = Field(min_length=len(CHOICES), max_length=len(CHOICES))
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("probabilities")
    @classmethod
    def valid_probabilities(cls, values):
        import math

        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Teacher probabilities must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Teacher probabilities must sum to one")
        return values


class TeacherResponse(BaseModel):
    revision: str = Field(min_length=1)
    items: list[TeacherLabel]


def canonical_input(entity, item, aliases=(), matching=None):
    """Keep every raw context character; no legacy 1200-character truncation."""
    if isinstance(item, BaseModel):
        item = item.model_dump()
    if not isinstance(item, Mapping):
        raise ValueError("Teacher input must be a comment mapping")
    if isinstance(matching, BaseModel):
        matching = matching.model_dump()
    matcher = MentionMatcher(str(entity).strip(), aliases, matching)
    return {
        "schema": INPUT_HASH_VERSION,
        "entity": str(entity).strip(),
        "aliases": list(aliases),
        "matching": matcher.options,
        "item": {
            "id": str(item.get("id") or ""),
            "story_id": str(item.get("story_id") or ""),
            "parent_id": None if item.get("parent_id") is None else str(item["parent_id"]),
            "story_title": str(item.get("story_title", item.get("title")) or ""),
            "parent_text": item.get("parent_text", item.get("parent")),
            "text": str(item.get("text") or ""),
        },
    }


def input_hash(entity, item, aliases=(), matching=None):
    payload = json.dumps(canonical_input(entity, item, aliases, matching), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# The explicit alias keeps client code readable beside other fingerprint types.
teacher_input_hash = input_hash
