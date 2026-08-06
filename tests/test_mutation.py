"""Unit tests for the mutation filter's parsers.

Only the string-processing parts run here; mutmut itself is Linux-only, so the
orchestration function is exercised live by the Colab notebook. These tests pin
the parsing behavior against the documented mutmut 3.x output format.
"""

import sys

import pytest

from llmtestgen.filters.mutation import (
    MutationResult,
    extract_survivors,
    parse_run_stats,
    run_mutation,
)

SAMPLE_RUN_OUTPUT = (
    "Running stats\n"
    "⠻ 10/253  \U0001F389 7 \U0001FAE5 0  ⏰ 0  \U0001F914 0  \U0001F641 3  \U0001F507 0\n"
    "⠹ 120/253  \U0001F389 101 \U0001FAE5 2  ⏰ 1  \U0001F914 0  \U0001F641 16  \U0001F507 0\n"
    "⣸ 253/253  \U0001F389 217 \U0001FAE5 4  ⏰ 2  \U0001F914 1  \U0001F641 27  \U0001F507 2\n"
)


def test_parse_run_stats_takes_the_final_line():
    result = parse_run_stats(SAMPLE_RUN_OUTPUT)
    assert (result.killed, result.no_tests, result.timeout) == (217, 4, 2)
    assert (result.suspicious, result.survived, result.skipped) == (1, 27, 2)
    assert result.total == 253


def test_parse_run_stats_handles_missing_stats():
    result = parse_run_stats("mutmut crashed before mutating anything")
    assert result.total == 0
    assert result.score is None
    assert result.no_active_tests is False
    assert "crashed" in result.raw_output


NO_ACTIVE_TESTS_OUTPUT = (
    "Running stats\n"
    "Stopping early, because we could not find any test case for any mutant. "
    "It seems that the selected tests do not cover any code that we mutated.\n"
    "You can set debug=true to see the executed test names in the output above.\n"
    "You can use mutmut browse to check which parts of the source code we mutated.\n"
)


def test_no_active_tests_early_exit_scores_zero():
    # mutmut 3.6 exits before any progress line when the suite runs green but
    # no test executes a mutated function (python-slugify's special.py: all
    # execution happens at import). Untested mutants count against the suite,
    # so this is a measured 0.0, not an instrument failure.
    result = parse_run_stats(NO_ACTIVE_TESTS_OUTPUT)
    assert result.no_active_tests is True
    assert result.total == 0
    assert result.score == 0.0


def test_no_active_tests_alternate_wording_scores_zero():
    result = parse_run_stats("failed to collect stats, no active tests found\n")
    assert result.no_active_tests is True
    assert result.score == 0.0


def test_real_stats_line_wins_over_marker_scan():
    # A run that produced a stats line is parsed normally even if the raw text
    # elsewhere happened to contain a marker-like phrase.
    result = parse_run_stats(SAMPLE_RUN_OUTPUT)
    assert result.no_active_tests is False
    assert result.total == 253


def test_score_convention():
    result = MutationResult(
        killed=8, timeout=1, survived=3, suspicious=1, no_tests=2, skipped=5
    )
    # detected = killed + timeout = 9; undetected = 3 + 1 + 2 = 6; skipped excluded.
    assert result.detected == 9
    assert result.undetected == 6
    assert result.score == pytest.approx(9 / 15)


def test_extract_survivors_status_per_line():
    text = (
        "sample_module.x_slugify__mutmut_1: killed\n"
        "sample_module.x_slugify__mutmut_2: survived\n"
        "sample_module.x_reduce_fraction__mutmut_7: survived\n"
    )
    assert extract_survivors(text) == [
        "sample_module.x_slugify__mutmut_2",
        "sample_module.x_reduce_fraction__mutmut_7",
    ]


def test_extract_survivors_section_listing():
    text = (
        "Survived \U0001F641 (2)\n"
        "    sample_module.x_slugify__mutmut_2\n"
        "    sample_module.x_counter__mutmut_4\n"
        "Killed \U0001F389 (1)\n"
        "    sample_module.x_slugify__mutmut_1\n"
    )
    assert extract_survivors(text) == [
        "sample_module.x_slugify__mutmut_2",
        "sample_module.x_counter__mutmut_4",
    ]


def test_run_mutation_refuses_native_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    module = tmp_path / "m.py"
    module.write_text("def f():\n    return 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Colab"):
        run_mutation("def test_f():\n    assert True\n", module)
