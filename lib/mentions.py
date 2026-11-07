"""Small, deterministic mention gate shared by collection and ML services.

Aliases and optional context hints belong to monitoring configuration. A story's
title never makes every comment in its thread relevant: context needs an
explicit reference in the comment and either a matching parent or a root reply.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Literal, Optional


MentionKind = Literal["direct", "alias", "typo", "context"]
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_PARENT_REFERENCE = re.compile(
    r"^(?:it|its|they|their|he|his|she|her)\b"
    r"|^(?:this|that|the) (?:app|application|assistant|bot|chatbot|device|"
    r"model|product|service|software|system|tool)\b"
    r"|^(?:same(?: here)?|agreed|i agree|exactly)\b"
)
_ROOT_REFERENCE = re.compile(
    r"^(?:it|its)\b"
    r"|^(?:this|that) (?:app|application|assistant|bot|chatbot|device|"
    r"model|product|service|software|system|tool)\b"
)


def _words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def _normalized(text: str) -> str:
    return " ".join(_words(text))


def _contains(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def _one_edit(left: str, right: str) -> bool:
    """Allow one insertion, deletion, substitution, or adjacent transposition."""
    if abs(len(left) - len(right)) > 1:
        return False
    if left == right:
        return True
    if len(left) == len(right):
        different = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(different) == 1:
            return True
        return (
            len(different) == 2
            and different[1] == different[0] + 1
            and left[different[0]] == right[different[1]]
            and left[different[1]] == right[different[0]]
        )
    short, long = (left, right) if len(left) < len(right) else (right, left)
    index = next((i for i, (a, b) in enumerate(zip(short, long)) if a != b), len(short))
    return short[index:] == long[index + 1:]


class MentionMatcher:
    """Match only the named entity, configured aliases, and obvious references.

    ``options`` is a JSON-serializable policy that can travel with a trained
    student artifact. ``match_text`` excludes contextual inference, so it is
    safe for checking a parent or title without following reply chains.
    """

    def __init__(
        self,
        entity: str,
        aliases: Sequence[str] = (),
        options: Optional[dict] = None,
    ):
        self.entity = entity
        self.aliases = list(aliases)
        supplied = options or {}
        self.options = {
            "allow_typos": bool(supplied.get("allow_typos", True)),
            "allow_context": bool(supplied.get("allow_context", True)),
            "context_hints": list(supplied.get("context_hints", [])),
        }
        self._entity = _normalized(entity)
        if not self._entity:
            raise ValueError("monitoring entity must contain a word")
        self._aliases = list(dict.fromkeys(
            normalized for alias in aliases
            if (normalized := _normalized(alias)) and normalized != self._entity
        ))
        self._hints = [_normalized(hint) for hint in self.options["context_hints"]]
        self._typo_names = list(dict.fromkeys(
            phrase.replace(" ", "")
            for phrase in [self._entity, *self._aliases]
            if len(phrase.replace(" ", "")) >= 6
        ))

    def match_text(self, text: str) -> Optional[MentionKind]:
        """Return direct/alias/typo; do not use a title or parent as evidence."""
        words = _words(text or "")
        normalized = " ".join(words)
        if _contains(normalized, self._entity):
            return "direct"
        if any(_contains(normalized, alias) for alias in self._aliases):
            return "alias"
        if not self.options["allow_typos"]:
            return None
        # A short window recognizes ChatGTP, ChaptGPT, and separated spellings
        # without searching arbitrary substrings inside unrelated longer words.
        for start in range(len(words)):
            candidate = ""
            for word in words[start:start + 3]:
                candidate += word
                if any(_one_edit(candidate, name) for name in self._typo_names):
                    return "typo"
        return None

    def has_text_match(self, text: str) -> bool:
        return self.match_text(text) is not None

    def match(self, item: object) -> Optional[MentionKind]:
        """Match a comment mapping or DTO using at most its immediate parent."""
        def value(name: str, default=None):
            return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)

        text = value("text", "") or ""
        direct = self.match_text(text)
        if direct is not None or not self.options["allow_context"]:
            return direct
        normalized = _normalized(text)
        if self.has_text_match(value("parent_text", "") or ""):
            if _PARENT_REFERENCE.search(normalized) or any(
                _contains(normalized, hint) for hint in self._hints
            ):
                return "context"
        parent_id, story_id = value("parent_id"), value("story_id")
        if (
            parent_id is not None
            and story_id is not None
            and str(parent_id) == str(story_id)
            and self.has_text_match(value("story_title", "") or "")
            and _ROOT_REFERENCE.search(normalized)
        ):
            return "context"
        return None
