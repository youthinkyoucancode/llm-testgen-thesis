"""Filter 4: mutation kill (Menzel condition 4).

Deliberately injects small bugs ("mutants") into the target module with mutmut
and counts how many the test suite catches. Coverage says which lines the tests
execute; the mutation score says whether the tests would notice if those lines
went wrong, which is the property a test suite exists for in the first place.

Platform note: mutmut 3.x refuses to run on native Windows, so ``run_mutation``
executes only on Linux (for this project: the Colab notebook). The parsers in
this file are pure string processing and are unit-tested locally on any OS; the
first Colab run validates them against real mutmut output.

Score convention, written down here because it goes in the methods chapter:
detected = killed + timeout (a mutant that makes the suite hang was caught),
undetected = survived + suspicious + no_tests (code no test exercises counts
against the suite), score = detected / (detected + undetected). Skipped mutants
are excluded. Raw counts are always recorded, so any other convention can be
recomputed from the logs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..target import TargetModule

# The one-line progress/stats format of `mutmut run` (mutmut 3.x), for example:
#   253/253  KILLED 217 NOTESTS 0  TIMEOUT 0  SUSPICIOUS 0  SURVIVED 36  SKIPPED 0
# rendered with emoji markers. Legend, per the mutmut docs:
#   [party] killed, [ghost] no tests, [clock] timeout, [thinking] suspicious,
#   [frown] survived, [mute] skipped.
_STATS = re.compile(
    r"(?P<done>\d+)/(?P<total>\d+)\s+"
    r"\U0001F389\s*(?P<killed>\d+)\s+"        # party popper: killed
    r"\U0001FAE5\s*(?P<no_tests>\d+)\s+"      # dotted-line face: no tests
    r"⏰\s*(?P<timeout>\d+)\s+"           # alarm clock: timeout
    r"\U0001F914\s*(?P<suspicious>\d+)\s+"    # thinking face: suspicious
    r"\U0001F641\s*(?P<survived>\d+)\s+"      # frown: survived
    r"\U0001F507\s*(?P<skipped>\d+)"          # muted speaker: skipped
)

# Mutant ids look like `sample_module.x_slugify__mutmut_3`.
_MUTANT_ID = re.compile(r"[A-Za-z_][\w.]*__mutmut_\d+")

# Status words that end a "survived" listing section (see extract_survivors).
_OTHER_STATUS = ("killed", "timed out", "timeout", "suspicious", "skipped", "no tests", "untested")


@dataclass
class MutationResult:
    """Counts from one mutmut run, plus surviving mutant ids for loop feedback."""

    killed: int = 0
    no_tests: int = 0
    timeout: int = 0
    suspicious: int = 0
    survived: int = 0
    skipped: int = 0
    survivors: list[str] = field(default_factory=list)
    raw_output: str = ""

    @property
    def total(self) -> int:
        return (
            self.killed + self.no_tests + self.timeout
            + self.suspicious + self.survived + self.skipped
        )

    @property
    def detected(self) -> int:
        return self.killed + self.timeout

    @property
    def undetected(self) -> int:
        return self.survived + self.suspicious + self.no_tests

    @property
    def score(self) -> float | None:
        """detected / (detected + undetected), or None when nothing was measured."""
        denominator = self.detected + self.undetected
        if denominator == 0:
            return None
        return self.detected / denominator


def parse_run_stats(output: str) -> MutationResult:
    """Parse the final stats line out of captured ``mutmut run`` output.

    The progress line repeats as mutmut works, so the LAST occurrence carries
    the final counts. Returns zeroed counts when no stats line is found (the
    caller can then inspect ``raw_output``).
    """
    last = None
    for match in _STATS.finditer(output):
        last = match
    if last is None:
        return MutationResult(raw_output=output)
    return MutationResult(
        killed=int(last["killed"]),
        no_tests=int(last["no_tests"]),
        timeout=int(last["timeout"]),
        suspicious=int(last["suspicious"]),
        survived=int(last["survived"]),
        skipped=int(last["skipped"]),
        raw_output=output,
    )


def extract_survivors(results_output: str) -> list[str]:
    """Pull surviving mutant ids out of ``mutmut results`` output.

    Tolerates both plausible shapes: a status printed on the same line as each
    mutant id, or a "Survived" section header followed by a list of ids. The
    first Colab run validates this against mutmut 3.6's real format.
    """
    survivors: list[str] = []
    in_survived_section = False
    for line in results_output.splitlines():
        ids = _MUTANT_ID.findall(line)
        low = line.lower()
        if "survived" in low or "\U0001F641" in line:
            survivors.extend(ids)
            in_survived_section = True
        elif any(word in low for word in _OTHER_STATUS):
            in_survived_section = False
        elif ids and in_survived_section:
            survivors.extend(ids)
    seen: set[str] = set()
    return [s for s in survivors if not (s in seen or seen.add(s))]


def _prepare_workspace(
    tests: str | Path, target: TargetModule, workspace: Path
) -> str:
    """Lay out a throwaway mutation workspace; return the path mutmut mutates.

    Single-file targets are copied to the workspace root (the original layout).
    Package targets get their whole top package copied in, so relative imports
    inside the target keep working while mutmut mutates only the one module.
    ``tests`` is generated test code (a string) or the path of an existing test
    file or directory (a library's human-written suite).
    """
    if target.top_package:
        shutil.copytree(
            target.import_root / target.top_package,
            workspace / target.top_package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        mutate_path = target.relative_file
    else:
        (workspace / target.path.name).write_text(
            target.path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        mutate_path = target.path.name

    tests_dir = workspace / "tests"
    if isinstance(tests, str):
        tests_dir.mkdir()
        (tests_dir / "test_generated.py").write_text(tests, encoding="utf-8")
    elif Path(tests).is_dir():
        shutil.copytree(
            Path(tests), tests_dir, ignore=shutil.ignore_patterns("__pycache__")
        )
    else:
        tests_dir.mkdir()
        source = Path(tests)
        (tests_dir / source.name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )

    (workspace / "pyproject.toml").write_text(
        "[tool.mutmut]\n"
        f'paths_to_mutate = ["{mutate_path}"]\n'
        'tests_dir = ["tests/"]\n',
        encoding="utf-8",
    )
    return mutate_path


def run_mutation(
    tests: str | Path,
    target_module: TargetModule | str | Path,
    *,
    timeout: float = 1800.0,
) -> MutationResult:
    """Run mutmut over ``target_module`` guarded by ``tests`` and score it.

    Builds a throwaway workspace (see ``_prepare_workspace``) so mutmut never
    touches the real tree, mirroring how ``execute_tests`` isolates pytest.
    Linux only; on native Windows this raises immediately with a pointer to the
    Colab notebook.
    """
    if sys.platform == "win32":
        raise RuntimeError(
            "mutmut 3.x does not run on native Windows. Run this stage in the "
            "Colab notebook (experiments/colab_runner.ipynb) or under WSL."
        )

    target = TargetModule.coerce(target_module)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        workspace = Path(tmp)
        _prepare_workspace(tests, target, workspace)

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{workspace}{os.pathsep}{existing}" if existing else str(workspace)

        try:
            run_proc = subprocess.run(
                [_mutmut_bin(), "run"],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return MutationResult(raw_output=f"mutmut exceeded the {timeout:.0f}s timeout")

        result = parse_run_stats((run_proc.stdout or "") + (run_proc.stderr or ""))

        results_proc = subprocess.run(
            [_mutmut_bin(), "results"],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        result.survivors = extract_survivors(results_proc.stdout or "")
        return result


def _mutmut_bin() -> str:
    """Path to the mutmut executable next to the running interpreter, else PATH."""
    candidate = Path(sys.executable).with_name("mutmut")
    if candidate.exists():
        return str(candidate)
    return "mutmut"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run mutation testing for one module against a test file or directory."
    )
    parser.add_argument("module", help="path to the target .py file")
    parser.add_argument("tests", help="path to the test file or directory holding the suite")
    parser.add_argument(
        "--import-root",
        default=None,
        help="directory that makes the module importable (for package targets, "
        "e.g. the sdist's src/); omit for a self-contained single file",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    module_target = TargetModule.from_path(args.module, args.import_root)
    outcome = run_mutation(Path(args.tests), module_target, timeout=args.timeout)

    if outcome.total == 0:
        print("warning: could not parse mutmut stats; raw output tail follows")
        tail = outcome.raw_output.splitlines()[-15:]
        for line in tail:
            print(f"  {line}")
    print(f"mutants  : {outcome.total} total")
    print(f"detected : {outcome.detected} (killed {outcome.killed}, timeout {outcome.timeout})")
    print(
        f"missed   : {outcome.undetected} (survived {outcome.survived}, "
        f"suspicious {outcome.suspicious}, untested {outcome.no_tests})"
    )
    print(f"skipped  : {outcome.skipped}")
    print(f"score    : {outcome.score:.3f}" if outcome.score is not None else "score    : n/a")
    if outcome.survivors:
        print("\nsurviving mutants (feedback for the refine prompt):")
        for mutant_id in outcome.survivors[:20]:
            print(f"  {mutant_id}")
        if len(outcome.survivors) > 20:
            print(f"  ... and {len(outcome.survivors) - 20} more")
