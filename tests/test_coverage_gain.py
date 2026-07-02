"""Tests for the coverage-gain filter."""

from pathlib import Path

from llmtestgen.filters.coverage_gain import measure_coverage

FIXTURE = Path(__file__).parent / "fixtures" / "sample_module.py"

PARTIAL = '''\
from sample_module import slugify


def test_slug():
    assert slugify("Hello World") == "hello-world"
'''

FULLER = '''\
import pytest

from sample_module import slugify, reduce_fraction, Counter


def test_slug():
    assert slugify("Hello World") == "hello-world"


def test_slug_no_lower():
    assert slugify("HELLO", lowercase=False) == "HELLO"


def test_reduce():
    assert reduce_fraction(98, 42) == (7, 3)


def test_reduce_zero():
    with pytest.raises(ZeroDivisionError):
        reduce_fraction(1, 0)


def test_counter():
    c = Counter()
    assert c.increment(3) == 3
    assert c.is_positive is True
'''


def test_measures_partial_coverage():
    result = measure_coverage(PARTIAL, FIXTURE)
    assert result.measured
    assert result.line_total > 0
    assert 0 < result.line_covered < result.line_total
    assert result.missing_lines  # slugify-only leaves reduce_fraction and Counter uncovered
    assert result.line_percent < 100


def test_more_tests_cover_more():
    partial = measure_coverage(PARTIAL, FIXTURE)
    fuller = measure_coverage(FULLER, FIXTURE)
    assert fuller.line_covered > partial.line_covered
    assert fuller.branch_total > 0  # branch measurement is on
    assert len(fuller.missing_lines) < len(partial.missing_lines)
