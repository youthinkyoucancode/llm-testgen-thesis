"""Fixture module inside a package, mirroring the real targets' shape.

The relative import below is the whole point: a target like ``click/types.py``
does exactly this, so any workspace that copies this module without its package
breaks immediately. Tests that pass against this file prove the package-aware
plumbing works.
"""

from __future__ import annotations

from ._helpers import normalize


def titlecase(text: str) -> str:
    """Title-case each word after normalizing whitespace; empty input gives ''."""
    cleaned = normalize(text)
    if not cleaned:
        return ""
    return " ".join(word[0].upper() + word[1:].lower() for word in cleaned.split(" "))


def word_count(text: str) -> int:
    """Number of whitespace-separated words after normalization."""
    cleaned = normalize(text)
    if not cleaned:
        return 0
    return len(cleaned.split(" "))
