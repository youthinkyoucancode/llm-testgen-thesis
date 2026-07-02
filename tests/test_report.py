"""Tests for the reporting module (pure, no model or subprocess)."""

import json

from llmtestgen.filters.coverage_gain import CoverageResult
from llmtestgen.generate import ModelConfig
from llmtestgen.loop import IterationRecord, PipelineResult
from llmtestgen.report import append_csv, build_record, write_json

MODEL = ModelConfig(provider="ollama", name="qwen2.5-coder", temperature=0.2, seed=42)
STAMP = "2026-07-01T10:00:00+00:00"


def make_result() -> PipelineResult:
    cov = CoverageResult(
        measured=True, line_covered=22, line_total=25,
        branch_covered=2, branch_total=4, missing_lines=[30, 37, 53],
    )
    iterations = [
        IterationRecord(index=0, kind="generate", prompt_version="v1_generate", tokens=500,
                        tests_kept=3, line_covered=18, branch_covered=1, line_percent=72.0, gain=True),
        IterationRecord(index=1, kind="refine", prompt_version="v1_refine", tokens=400,
                        tests_kept=5, line_covered=22, branch_covered=2, line_percent=88.0, gain=True),
    ]
    final_tests = "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n"
    return PipelineResult(final_tests=final_tests, final_coverage=cov, iterations=iterations,
                          total_tokens=900, stop_reason="no coverage gain")


def test_build_record_fields():
    rec = build_record(make_result(), module="sample_module.py", model=MODEL,
                       condition="C", timestamp=STAMP)
    assert rec.condition == "C"
    assert rec.model_name == "qwen2.5-coder"
    assert rec.prompt_versions == ["v1_generate", "v1_refine"]
    assert rec.iterations == 2
    assert rec.tests_retained == 2
    assert (rec.line_covered, rec.line_total, rec.line_percent) == (22, 25, 88.0)
    assert rec.stop_reason == "no coverage gain"
    assert rec.mutation_score is None


def test_write_json_roundtrip(tmp_path):
    rec = build_record(make_result(), module="m.py", model=MODEL, condition="C", timestamp=STAMP)
    path = write_json(rec, tmp_path / "results" / "m.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["module"] == "m.py"
    assert len(data["per_iteration"]) == 2
    assert data["per_iteration"][0]["kind"] == "generate"


def test_append_csv_writes_header_once(tmp_path):
    rec = build_record(make_result(), module="m.py", model=MODEL, condition="C", timestamp=STAMP)
    csv_path = tmp_path / "summary.csv"
    append_csv(rec, csv_path)
    append_csv(rec, csv_path)
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # one header plus two rows
    assert lines[0].startswith("timestamp,module,condition")
