"""Tests for the campaign runner's redo path.

The mutation fill introduced after the 2026-08-11 run must reuse the banked
suite and coverage verbatim; these tests pin the record round trip and the
redo rule the fill depends on. The runner script is not a package, so it is
imported off the experiments directory the same way it arranges its own path.
"""

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from run_experiments import needs_mutation_redo, record_from_banked  # noqa: E402

from llmtestgen.report import ExperimentRecord  # noqa: E402


def make_record(**overrides) -> ExperimentRecord:
    base = dict(
        module="sample.module", condition="C", model_provider="ollama",
        model_name="qwen2.5-coder", temperature=0.2, seed=42,
        prompt_versions=["v3"], iterations=2, tests_retained=4,
        line_covered=40, line_total=50, line_percent=80.0,
        branch_covered=10, branch_total=16, branch_percent=62.5,
        mutation_score=None, total_tokens=12345, stop_reason="max_iterations",
        timestamp="2026-08-11T20:00:00+00:00",
        per_iteration=[{"prompt_version": "v3"}],
        final_tests="def test_a():\n    assert True\n",
    )
    base.update(overrides)
    return ExperimentRecord(**base)


def test_record_round_trips_through_banked_json():
    record = make_record()
    assert record_from_banked(asdict(record)) == record


def test_unknown_banked_keys_are_dropped():
    banked = asdict(make_record())
    banked["added_in_a_future_version"] = 123
    assert record_from_banked(banked) == make_record()


def test_redo_rule_targets_only_unscored_records_with_coverage():
    covered_unscored = asdict(make_record())
    covered_scored = asdict(make_record(mutation_score=0.0))
    uncovered = asdict(make_record(line_percent=0.0))

    assert needs_mutation_redo(covered_unscored, skip_mutation=False) is True
    # A measured score, including 0.0, is final: the redo loop must terminate.
    assert needs_mutation_redo(covered_scored, skip_mutation=False) is False
    # An empty-suite record never gets a mutation score; it stays banked.
    assert needs_mutation_redo(uncovered, skip_mutation=False) is False
    # Coverage-only passes never trigger rebuilds.
    assert needs_mutation_redo(covered_unscored, skip_mutation=True) is False
