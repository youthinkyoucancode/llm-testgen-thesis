"""Tests for the execute-check filter."""

from pathlib import Path

from llmtestgen.filters.execute import execute_tests, keep_passing

FIXTURE = Path(__file__).parent / "fixtures" / "sample_module.py"

# Two correct tests and one with a wrong assertion, mirroring the mistake a
# single-pass LLM makes (here: expecting slugify to keep punctuation and to
# lowercase even though lowercase=False).
MIXED = '''\
from sample_module import slugify, reduce_fraction


def test_slug_ok():
    assert slugify("Hello World") == "hello-world"


def test_reduce_ok():
    assert reduce_fraction(98, 42) == (7, 3)


def test_slug_wrong():
    assert slugify("Special!", lowercase=False) == "special!"
'''

IMPORT_ERROR = '''\
from sample_module import does_not_exist


def test_x():
    assert True
'''


def test_reports_pass_and_fail():
    result = execute_tests(MIXED, FIXTURE)
    assert result.collected
    assert {o.name for o in result.passed} == {"test_slug_ok", "test_reduce_ok"}
    assert {o.name for o in result.failed} == {"test_slug_wrong"}
    assert result.failed[0].message  # carries the assertion message for feedback


def test_keep_passing_drops_only_the_failing_test():
    result = execute_tests(MIXED, FIXTURE)
    pruned = keep_passing(MIXED, result)
    assert "test_slug_ok" in pruned
    assert "test_reduce_ok" in pruned
    assert "test_slug_wrong" not in pruned
    # the pruned suite should now be all-green
    assert execute_tests(pruned, FIXTURE).all_passed


def test_import_error_yields_no_passes():
    result = execute_tests(IMPORT_ERROR, FIXTURE)
    assert not result.any_passed
