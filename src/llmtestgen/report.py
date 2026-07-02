"""Per-run reporting.

Turns a PipelineResult into a structured, reproducible experiment record and
writes it as JSON (full detail, including the per-iteration log) and as a CSV row
(flat summary for aggregate analysis in pandas). Every record carries the model
id, parameters, prompt versions, and coverage so a run can be reproduced and
compared across the three conditions.
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .generate import ModelConfig
from .loop import PipelineResult


@dataclass
class ExperimentRecord:
    module: str
    condition: str
    model_provider: str
    model_name: str
    temperature: float
    seed: int | None
    prompt_versions: list[str]
    iterations: int
    tests_retained: int
    line_covered: int
    line_total: int
    line_percent: float
    branch_covered: int
    branch_total: int
    branch_percent: float
    mutation_score: float | None
    total_tokens: int
    stop_reason: str
    timestamp: str
    per_iteration: list[dict] = field(default_factory=list)


# Columns written to the flat CSV; the nested per-iteration log stays in the JSON only.
_CSV_FIELDS = [
    "timestamp", "module", "condition", "model_provider", "model_name",
    "temperature", "seed", "iterations", "tests_retained",
    "line_covered", "line_total", "line_percent",
    "branch_covered", "branch_total", "branch_percent",
    "mutation_score", "total_tokens", "stop_reason",
]


def build_record(
    result: PipelineResult,
    *,
    module: str,
    model: ModelConfig,
    condition: str,
    mutation_score: float | None = None,
    timestamp: str | None = None,
) -> ExperimentRecord:
    """Assemble an ExperimentRecord from a finished pipeline run."""
    cov = result.final_coverage
    versions: list[str] = []
    for record in result.iterations:
        if record.prompt_version not in versions:
            versions.append(record.prompt_version)

    return ExperimentRecord(
        module=str(module),
        condition=condition,
        model_provider=model.provider,
        model_name=model.name,
        temperature=model.temperature,
        seed=model.seed,
        prompt_versions=versions,
        iterations=len(result.iterations),
        tests_retained=_count_tests(result.final_tests),
        line_covered=cov.line_covered if cov else 0,
        line_total=cov.line_total if cov else 0,
        line_percent=round(cov.line_percent, 2) if cov else 0.0,
        branch_covered=cov.branch_covered if cov else 0,
        branch_total=cov.branch_total if cov else 0,
        branch_percent=round(cov.branch_percent, 2) if cov else 0.0,
        mutation_score=mutation_score,
        total_tokens=result.total_tokens,
        stop_reason=result.stop_reason,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        per_iteration=[asdict(r) for r in result.iterations],
    )


def write_json(record: ExperimentRecord, path: str | Path) -> Path:
    """Write the full record (including per-iteration log) to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return path


def append_csv(record: ExperimentRecord, path: str | Path) -> Path:
    """Append the flat summary of a record to a CSV, writing the header if the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in asdict(record).items() if k in _CSV_FIELDS}
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return path


def _count_tests(code: str) -> int:
    """Count top-level test functions (names starting with 'test') in the suite."""
    if not code.strip():
        return 0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    return sum(
        1 for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
    )
