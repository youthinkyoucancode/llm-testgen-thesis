"""The pre-registered analysis: pairing, imputation, Holm, rank-biserial.

Statistical helpers are checked against hand-computed examples; the full run
is smoke-tested over a synthetic results directory shaped exactly like the
campaign's output.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_results", Path(__file__).resolve().parents[1] / "experiments" / "analyze_results.py"
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("analyze_results", _MOD)
_SPEC.loader.exec_module(_MOD)


def record(module, condition, seed=None, line=0.0, branch=0.0, mutation=None, retained=0):
    return {
        "module": module, "condition": condition, "seed": seed,
        "line_percent": line, "branch_percent": branch,
        "mutation_score": mutation, "tests_retained": retained,
        "total_tokens": 1000,
    }


# --- the pre-registered rules, unit by unit --------------------------------

def test_empty_suite_scores_zero_mutation():
    assert _MOD.effective_mutation(record("m", "B", retained=0, mutation=None)) == 0.0


def test_nonempty_suite_without_score_is_excluded_not_imputed():
    assert _MOD.effective_mutation(record("m", "A", retained=5, mutation=None)) is None


def test_recorded_scores_pass_through():
    assert _MOD.effective_mutation(record("m", "C", retained=3, mutation=0.417)) == 0.417


def test_failed_condition_a_measurement_is_never_imputed_to_zero():
    # Condition A leaves tests_retained at 0 because the field counts tests the
    # PIPELINE retained, and condition A generates nothing. Reading that as an
    # empty suite would report a mature human suite as killing no mutants. On
    # the 2026-08-13 data this affected seven of twelve condition-A records.
    failed = record("markdown.blockprocessors", "A", retained=0, mutation=None, line=97.74)
    assert _MOD.effective_mutation(failed) is None


def test_condition_a_measured_zero_still_passes_through():
    # slugify.special's human suite genuinely scores 0.0: it exercises the
    # module only at import, so mutmut finds no test covering any mutant. A
    # recorded 0.0 is a measurement and must survive.
    measured = record("slugify.special", "A", retained=0, mutation=0.0, line=100.0)
    assert _MOD.effective_mutation(measured) == 0.0


def test_seed_median_is_the_median_across_seeds():
    records = {42: record("m", "C", 42, line=60.0), 43: record("m", "C", 43, line=70.0),
               44: record("m", "C", 44, line=0.0)}
    value, notes = _MOD.seed_median(records, "line_percent")
    assert value == 60.0
    assert notes == []


def test_holm_bonferroni_known_example():
    # sorted p: 0.01, 0.03, 0.04 with m=3 -> 0.03, 0.06, then max(0.04, 0.06)=0.06
    adjusted = _MOD.holm_bonferroni([0.01, 0.04, 0.03])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.06)
    assert adjusted[2] == pytest.approx(0.06)


def test_holm_passes_none_through():
    adjusted = _MOD.holm_bonferroni([0.02, None])
    assert adjusted[0] == pytest.approx(0.02)  # m=1: only defined tests count
    assert adjusted[1] is None


def test_rank_biserial_hand_example():
    # diffs (C-B): +1, +2, +3, -1 -> |d| ranks 1.5, 3, 4, 1.5
    # T+ = 8.5, T- = 1.5, r = 7/10 = 0.7
    pairs = [("a", 0, 1), ("b", 0, 2), ("c", 0, 3), ("d", 1, 0)]
    result = _MOD.paired_test(pairs)
    assert result["rank_biserial"] == pytest.approx(0.7)
    assert result["n_nonzero"] == 4


def test_paired_test_with_no_nonzero_differences_is_undefined():
    pairs = [("a", 1.0, 1.0), ("b", 2.0, 2.0)]
    result = _MOD.paired_test(pairs)
    assert result["p"] is None and result["W"] is None
    assert result["n_nonzero"] == 0


# --- the full run over a synthetic campaign --------------------------------

def _write(results_dir, rec, stem):
    (results_dir / f"{stem}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_analyze_end_to_end(tmp_path):
    for module, ceiling in [("libx.alpha", 90.0), ("libx.beta", 95.0)]:
        _write(tmp_path, record(module, "A", line=ceiling, branch=80.0,
                                mutation=0.8, retained=20), f"{module}_A")
        for seed in (42, 43, 44):
            _write(tmp_path, record(module, "B", seed, line=0.0, retained=0),
                   f"{module}_B_s{seed}")
            _write(tmp_path, record(module, "C", seed, line=60.0 + seed % 3,
                                    branch=50.0, mutation=0.4, retained=5),
                   f"{module}_C_s{seed}")
    outcome = _MOD.analyze(tmp_path)

    assert len(outcome["modules"]) == 2
    assert outcome["exclusions"] == []
    line_test = outcome["tests"]["line_percent"]
    assert line_test["n_pairs"] == 2
    assert line_test["median_B"] == 0.0
    assert line_test["rank_biserial"] == pytest.approx(1.0)   # C wins every pair
    mutation_test = outcome["tests"]["mutation_score"]
    assert mutation_test["median_B"] == 0.0                    # imputed, not dropped
    assert mutation_test["n_pairs"] == 2
    for name in ("per_module.csv", "per_module.md", "paired_tests.csv",
                 "paired_tests.md", "per_seed.csv"):
        assert (tmp_path / "analysis" / name).exists()


def test_analyze_excludes_instrument_failures(tmp_path):
    _write(tmp_path, record("libx.alpha", "A", line=90.0, mutation=None, retained=20),
           "libx.alpha_A")
    for seed in (42, 43, 44):
        _write(tmp_path, record("libx.alpha", "B", seed, line=0.0, retained=0),
               f"libx.alpha_B_s{seed}")
        _write(tmp_path, record("libx.alpha", "C", seed, line=60.0, mutation=None,
                                retained=5), f"libx.alpha_C_s{seed}")
    outcome = _MOD.analyze(tmp_path)
    assert any("without mutation score" in line for line in outcome["exclusions"])
    assert outcome["tests"]["mutation_score"]["n_pairs"] == 0
    assert (tmp_path / "analysis" / "exclusions.txt").exists()
