"""Shared helpers for scenario-aware grader keyword matching."""
from __future__ import annotations

from typing import Iterable


def normalize_text(value: object) -> str:
    """Normalize free-form answers and scenario keywords for tolerant matching."""
    return " ".join(
        str(value)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def contains_term(text: str, term: object) -> bool:
    normalized = normalize_text(term)
    if not normalized:
        return False
    if normalized in text:
        return True
    return normalized.replace(" ", "") in text.replace(" ", "")


def contains_any(text: str, terms: Iterable[object]) -> bool:
    return any(contains_term(text, term) for term in terms if term is not None)


def breakdown_label(value: object) -> str:
    return normalize_text(value).replace(" ", "_")
