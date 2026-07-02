"""A small sample module used to exercise the pipeline on known input.

It deliberately mixes shapes the extractor must handle: type hints, keyword-only
arguments, an exception path, a private helper, a class with ``__init__``, a
method with a default argument, and a decorated property.
"""

from __future__ import annotations

import re
from math import gcd


def slugify(text: str, *, lowercase: bool = True) -> str:
    """Turn arbitrary text into a URL-safe slug.

    Collapses runs of whitespace and punctuation into single hyphens and trims
    hyphens from the ends.
    """
    text = text.strip()
    if lowercase:
        text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text).strip("-")


def reduce_fraction(numerator: int, denominator: int) -> tuple[int, int]:
    """Reduce a fraction to lowest terms. Raises ZeroDivisionError if denominator is 0."""
    if denominator == 0:
        raise ZeroDivisionError("denominator must be non-zero")
    divisor = gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def _internal_helper(x: int) -> int:
    # Private by convention; the extractor should mark it non-public.
    return x * 2


class Counter:
    """A tiny stateful counter."""

    def __init__(self, start: int = 0) -> None:
        self.value = start

    def increment(self, by: int = 1) -> int:
        """Add ``by`` to the counter and return the new value."""
        self.value += by
        return self.value

    @property
    def is_positive(self) -> bool:
        return self.value > 0
