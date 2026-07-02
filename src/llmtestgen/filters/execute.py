"""Filter 2: execute check.

Runs candidate tests against the *unmodified* target module in an isolated
subprocess and reports which tests pass. A generated test that fails is almost
always a wrong assertion, not a real bug, so the pipeline keeps the passing
tests, drops the failing ones, and can feed their error messages back into the
refinement loop. Running in a subprocess means bad generated code cannot crash
the pipeline, and a timeout guards against tests that hang.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestOutcome:
    name: str
    passed: bool
    message: str = ""      # failure or error message when the test did not pass


@dataclass
class ExecuteResult:
    collected: bool                       # pytest ran and produced a parseable report
    outcomes: list[TestOutcome] = field(default_factory=list)
    timed_out: bool = False
    returncode: int | None = None
    raw_output: str = ""

    @property
    def passed(self) -> list[TestOutcome]:
        return [o for o in self.outcomes if o.passed]

    @property
    def failed(self) -> list[TestOutcome]:
        return [o for o in self.outcomes if not o.passed]

    @property
    def any_passed(self) -> bool:
        return bool(self.passed)

    @property
    def all_passed(self) -> bool:
        return self.collected and bool(self.outcomes) and not self.failed


def execute_tests(
    test_code: str,
    target_module: str | Path,
    *,
    timeout: float = 60.0,
) -> ExecuteResult:
    """Write ``test_code`` to a temp file and run it with pytest against ``target_module``.

    The target module's directory is put on ``PYTHONPATH`` so the generated
    ``from <module> import ...`` line resolves. Per-test outcomes are read back
    from a JUnit XML report, which needs no extra pytest plugin.
    """
    target_module = Path(target_module).resolve()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        test_file = tmp_path / "test_candidate.py"
        test_file.write_text(test_code, encoding="utf-8")
        report = tmp_path / "report.xml"

        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            f"--junitxml={report}",
            "-q",
            "-p", "no:cacheprovider",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp_path,
                env=_env_with_module_on_path(target_module.parent),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResult(
                collected=False,
                timed_out=True,
                raw_output=f"tests exceeded the {timeout:.0f}s timeout",
            )

        if not report.exists():
            # No report means pytest crashed before running anything (e.g. a usage error).
            return ExecuteResult(
                collected=False,
                returncode=proc.returncode,
                raw_output=(proc.stdout + proc.stderr).strip(),
            )

        outcomes = _parse_junit(report)
        return ExecuteResult(
            collected=bool(outcomes),
            outcomes=outcomes,
            returncode=proc.returncode,
            raw_output=proc.stdout.strip(),
        )


def keep_passing(test_code: str, result: ExecuteResult) -> str:
    """Return ``test_code`` with the failing top-level test functions removed."""
    failing = {o.name for o in result.failed}
    if not failing:
        return test_code

    tree = ast.parse(test_code)
    drop: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in failing:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            drop.update(range(start, (node.end_lineno or start) + 1))

    lines = test_code.splitlines(keepends=True)
    kept = [line for i, line in enumerate(lines, start=1) if i not in drop]
    return "".join(kept).strip() + "\n"


def _env_with_module_on_path(module_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{module_dir}{os.pathsep}{existing}" if existing else str(module_dir)
    return env


def _parse_junit(report_path: Path) -> list[TestOutcome]:
    """Read per-test outcomes from a pytest JUnit XML report."""
    root = ET.parse(report_path).getroot()
    outcomes: list[TestOutcome] = []
    for testcase in root.iter("testcase"):
        problem = testcase.find("failure")
        if problem is None:
            problem = testcase.find("error")
        if problem is None:
            outcomes.append(TestOutcome(name=testcase.get("name", "?"), passed=True))
        else:
            message = (problem.get("message") or problem.text or "").strip()
            first_line = message.splitlines()[0] if message else ""
            outcomes.append(TestOutcome(name=testcase.get("name", "?"), passed=False, message=first_line))
    return outcomes
