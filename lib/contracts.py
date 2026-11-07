"""Validated comment and prediction contracts shared by the HTTP services."""

import math
from datetime import datetime, timezone
from typing import Annotated, Optional

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

from .sentiment import CHOICES, OFF_TOPIC, SCORE_MAX, SCORE_MIN


Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Probability = Annotated[float, Field(ge=0, le=1)]
SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ProviderName = Annotated[
    str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_-]*$")
]


class MentionMatchingConfig(BaseModel):
    allow_typos: bool = True
    allow_context: bool = True
    context_hints: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    ] = Field(default_factory=list, max_length=20)


class MonitoringConfig(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    aliases: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]] = Field(
        default_factory=list, max_length=10
    )
    source: ProviderName
    matching: MentionMatchingConfig = Field(default_factory=MentionMatchingConfig)


class ProjectConfig(BaseModel):
    monitoring: MonitoringConfig
    offline_demo: bool
    path_to_offline_demo_data: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Comment(BaseModel):
    id: Identifier
    story_id: Identifier
    parent_id: Optional[Identifier] = None
    published_at: datetime
    story_title: str = Field(max_length=4096)
    text: str = Field(min_length=1, max_length=65536)
    parent_text: Optional[str] = Field(default=None, max_length=65536)
    url: str = Field(min_length=1, max_length=4096)
    source: Identifier
    fixture_label: Optional[int] = Field(default=None, ge=0, le=OFF_TOPIC)

    @field_validator("published_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("published_at must include a timezone")
        return value.astimezone(timezone.utc)


class Probabilities(BaseModel):
    comment_id: Identifier
    probabilities: list[Probability] = Field(min_length=len(CHOICES), max_length=len(CHOICES))
    content_hash: Optional[SHA256] = None

    @field_validator("probabilities")
    @classmethod
    def check_sum(cls, values: list[float]) -> list[float]:
        if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=0.001):
            raise ValueError("probabilities must sum to 1")
        return values


class Prediction(Probabilities):
    score: Optional[float] = Field(default=None, ge=SCORE_MIN, le=SCORE_MAX)
    relevant: bool

    @model_validator(mode="after")
    def check_score(self) -> "Prediction":
        if self.relevant != (self.score is not None):
            raise ValueError("score must be present exactly when relevant is true")
        return self
