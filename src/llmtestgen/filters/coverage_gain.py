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
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..target import TargetModule


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
    tests: str | Path,
    target_module: TargetModule | str | Path,
    *,
    branch: bool = True,
    timeout: float = 120.0,
) -> CoverageResult:
    """Run tests under coverage.py and report coverage of the target module.

    ``tests`` is either the test code itself (a string, written to a temp file)
    or the path of an existing test file or directory (how condition A runs a
    library's human-written suite). Collection is limited to the target's
    package (or its directory, for a single-file target), and the report is
    filtered down to the one module under test.
    """
    target = TargetModule.coerce(target_module)
    source_dir = (
        target.import_root / target.top_package
        if target.top_package
        else target.path.parent
    )
    env = _env_with_module_on_path(target.import_root)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        if isinstance(tests, str):
            test_target = tmp_path / "test_candidate.py"
            test_target.write_text(tests, encoding="utf-8")
        else:
            # Stage path-based suites into the neutral workspace instead of
            # running them in place: pytest puts the test file's own directory
            # first on sys.path, and a sibling copy of the package next to the
            # suite (a repo checkout ships one) would shadow the target on
            # PYTHONPATH, leaving coverage watching files that never import.
            source = Path(tests).resolve()
            staged = tmp_path / "tests"
            if source.is_dir():
                shutil.copytree(
                    source, staged, ignore=shutil.ignore_patterns("__pycache__")
                )
                test_target = staged
            else:
                staged.mkdir()
                test_target = staged / source.name
                test_target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        data_file = tmp_path / ".coverage"
        json_file = tmp_path / "coverage.json"

        run_cmd = [
            sys.executable, "-m", "coverage", "run",
            f"--data-file={data_file}",
            f"--source={source_dir}",
        ]
        if branch:
            run_cmd.append("--branch")
        run_cmd += [
            "-m", "pytest", str(test_target), "-q", "-p", "no:cacheprovider",
            # A real library's suite may contain a module pytest refuses to
            # collect (markdown ships a helper class named TestSuite that has an
            # __init__, which pytest reports as a collection error). By default
            # ONE such module aborts the entire run before any test executes,
            # and coverage then records only import-time statements: markdown's
            # condition A looked like 21-28% coverage when the true figure is
            # ~98%, and md_in_html looked like 0%. Continuing past collection
            # errors can only ADD executed tests, never remove them, so it makes
            # the measurement strictly more faithful to what the suite covers.
            "--continue-on-collection-errors",
        ]

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

        return _parse_coverage_json(json_file, target.relative_file, raw_output=run.stdout.strip())


def _parse_coverage_json(json_file: Path, relative_file: str, *, raw_output: str = "") -> CoverageResult:
    """Pick the target module out of the coverage report by path suffix.

    Suffix matching (never bare stem matching) keeps same-named files apart:
    click and jinja2 both ship a ``parser.py``, and ``test_x.py`` must not match
    a target ``x.py``.
    """
    data = json.loads(json_file.read_text(encoding="utf-8"))
    for path, info in data.get("files", {}).items():
        normalized = Path(path).as_posix()
        if normalized == relative_file or normalized.endswith("/" + relative_file):
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
