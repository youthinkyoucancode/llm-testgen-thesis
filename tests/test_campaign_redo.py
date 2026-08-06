"""The campaign's skip-or-redo decision for banked records.

Loaded from the experiments directory by file path; the campaign scripts are
not a package on purpose (they run on Colab from the repo root).
"""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_experiments", Path(__file__).resolve().parents[1] / "experiments" / "run_experiments.py"
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("run_experiments", _MOD)
_SPEC.loader.exec_module(_MOD)

needs_mutation_redo = _MOD.needs_mutation_redo


def test_complete_record_stays_skipped():
    record = {"line_percent": 92.4, "mutation_score": 0.738}
    assert not needs_mutation_redo(record, skip_mutation=False)


def test_covered_suite_without_mutation_score_reruns():
    # The 2026-08-04 failure shape: coverage measured, mutmut collected nothing.
    record = {"line_percent": 92.4, "mutation_score": None}
    assert needs_mutation_redo(record, skip_mutation=False)


def test_empty_suite_stays_skipped():
    # Condition B producing no surviving tests is a valid result, not a redo.
    record = {"line_percent": 0.0, "mutation_score": None}
    assert not needs_mutation_redo(record, skip_mutation=False)


def test_skip_mutation_pass_never_redoes():
    record = {"line_percent": 92.4, "mutation_score": None}
    assert not needs_mutation_redo(record, skip_mutation=True)


def test_unreadable_record_defaults_to_skip():
    assert not needs_mutation_redo({}, skip_mutation=False)


def test_measured_zero_score_is_final_not_redone():
    # The no-active-tests outcome (e.g. slugify.special's human suite, which
    # exercises the module only at import time) scores 0.0 by convention. The
    # redo loop must treat that as a result, or it would rebuild the record
    # on every session forever.
    record = {"line_percent": 100.0, "mutation_score": 0.0}
    assert not needs_mutation_redo(record, skip_mutation=False)
