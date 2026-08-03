"""Private helper module, imported relatively by textutils."""

from __future__ import annotations


def normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends."""
    return " ".join(text.split())
