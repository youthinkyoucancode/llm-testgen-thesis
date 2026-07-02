"""Filter 3: coverage gain.

Measures how much of the target module a test suite exercises, using coverage.py
in an isolated subprocess. It reports line and branch coverage and, most usefully
for the refinement loop, the list of *uncovered* lines and branches. Those
uncovered locations become the feedback that tells the model what to target in
the next round, and comparing coverage between rounds is how the loop decides
whether an iteration made progress (Menzel's stop-on-no-gain condition).

Measure this on the passing suite (after the execute filter has dropped failing
tests), so the numbers reflect only the tests the pipeline actually keeps.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CoverageResult:
    measured: bool
    line_covered: int = 0
    line_total: int = 0
    branch_covered: int = 0
    branch_total: int = 0
    missing_lines: list[int] = field(default_factory=list)
    missing_branches: list[list[int]] = field(default_factory=list)
    raw_output: str = ""

    @property
    def line_percent(self) -> float:
        return 100.0 * self.line_covered / self.line_total if self.line_total else 100.0

    @property
    def branch_percent(self) -> float:
        return 100.0 * self.branch_covered / self.branch_total if self.branch_total else 100.0


def measure_coverage(
    test_code: str,
    target_module: str | Path,
    *,
    branch: bool = True,
    timeout: float = 120.0,
) -> CoverageResult:
    """Run ``test_code`` under coverage.py against ``target_module`` and report coverage."""
    target_module = Path(target_module).resolve()
    module_name = target_module.stem
    env = _env_with_module_on_path(target_module.parent)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        test_file = tmp_path / "test_candidate.py"
        test_file.write_text(test_code, encoding="utf-8")
        data_file = tmp_path / ".coverage"
        json_file = tmp_path / "coverage.json"

        run_cmd = [
            sys.executable, "-m", "coverage", "run",
            f"--data-file={data_file}",
            f"--source={target_module.parent}",
        ]
        if branch:
            run_cmd.append("--branch")
        run_cmd += ["-m", "pytest", str(test_file), "-q", "-p", "no:cacheprovider"]

        try:
            run = subprocess.run(run_cmd, cwd=tmp_path, env=env,
                                 capture_output=True, text=True, timeout=timeout)
            subprocess.run(
                [sys.executable, "-m", "coverage", "json",
                 f"--data-file={data_file}", "-o", str(json_file)],
                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return CoverageResult(measured=False, raw_output=f"coverage exceeded the {timeout:.0f}s timeout")

        if not json_file.exists():
            return CoverageResult(measured=False, raw_output=(run.stdout + run.stderr).strip())

        return _parse_coverage_json(json_file, module_name, raw_output=run.stdout.strip())


def _parse_coverage_json(json_file: Path, module_name: str, *, raw_output: str = "") -> CoverageResult:
    data = json.loads(json_file.read_text(encoding="utf-8"))
    for path, info in data.get("files", {}).items():
        if Path(path).stem == module_name:
            summary = info["summary"]
            return CoverageResult(
                measured=True,
                line_covered=summary.get("covered_lines", 0),
                line_total=summary.get("num_statements", 0),
                branch_covered=summary.get("covered_branches", 0),
                branch_total=summary.get("num_branches", 0),
                missing_lines=info.get("missing_lines", []),
                missing_branches=info.get("missing_branches", []),
                raw_output=raw_output,
            )
    return CoverageResult(measured=False, raw_output=raw_output or "target module not found in coverage data")


def _env_with_module_on_path(module_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{module_dir}{os.pathsep}{existing}" if existing else str(module_dir)
    return env
