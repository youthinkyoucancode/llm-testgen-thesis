"""Tests for the parse-check filter."""

from llmtestgen.filters.parse_check import extract_code, parse_check

FENCED_GOOD = """Here are the tests:
```python
import pytest


def test_add():
    assert 1 + 1 == 2
```
"""

FENCED_BAD = """```python
def test_broken(:
    pass
```"""


def test_extracts_code_from_fence():
    code = extract_code(FENCED_GOOD)
    assert code.startswith("import pytest")
    assert "def test_add" in code
    assert "```" not in code


def test_valid_code_passes():
    result = parse_check(FENCED_GOOD)
    assert result.ok
    assert result.error is None


def test_broken_code_fails_with_message():
    result = parse_check(FENCED_BAD)
    assert not result.ok
    assert result.error  # a syntax-error message


def test_unfenced_text_is_treated_as_code():
    result = parse_check("def test_x():\n    assert True\n")
    assert result.ok
