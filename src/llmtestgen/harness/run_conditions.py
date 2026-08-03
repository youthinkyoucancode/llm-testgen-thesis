"""Experiment harness: run the three conditions on a module and record each.

- A (reference ceiling): the module's existing human-written test suite.
- B (primary baseline): single-pass generation, no refinement (the loop capped
  at one iteration, with no coverage-driven feedback).
- C (treatment): the full iterative pipeline.

B and C are the *same* ``run_pipeline`` with a different iteration cap, so the
only thing that differs between baseline and treatment is the refinement loop
itself, which keeps the comparison fair by construction. Mutation score is added
on Colab (mutmut is Linux-only); here the harness records coverage plus the run
metadata for every condition.
"""

from __future__ import annotations

from pathlib import Path

from ..filters.coverage_gain import measure_coverage
from ..generate import ModelConfig, generate
from ..loop import LoopConfig, PipelineResult, run_pipeline
from ..report import ExperimentRecord, append_csv, build_record, write_json
from ..target import TargetModule

# Condition A uses no model; this sentinel makes that explicit in every record.
_HUMAN = ModelConfig(provider="human", name="human-written", temperature=0.0, seed=None)


def run_condition_a(
    module: TargetModule | str | Path, human_tests: str | Path, *,
    timestamp: str | None = None, coverage_timeout: float = 600.0,
) -> ExperimentRecord:
    """Measure the existing human-written suite (the reference ceiling).

    ``human_tests`` is the path of the suite: a single test file (the fixture
    setup) or a whole tests directory (a real library's suite, conftest and
    all). Coverage is measured of the target module only, whichever tests in
    that suite happen to exercise it. ``tests_retained`` is counted only for
    single-file suites; for a directory it records 0, and the A-metrics that
    matter are coverage and mutation score.
    """
    target = TargetModule.coerce(module)
    tests_path = Path(human_tests)
    if not tests_path.exists():
        # A missing suite must fail loudly: recording it as 0% coverage would
        # silently poison the reference ceiling (learned from python-slugify,
        # whose sdist ships no tests at all).
        raise FileNotFoundError(f"human test suite not found: {tests_path}")
    cov = measure_coverage(tests_path, target, timeout=coverage_timeout)
    if not cov.measured:
        # Usually a timeout (click's suite collects 32k+ tests); the fix is a
        # bigger coverage_timeout, and the raw output says so.
        raise RuntimeError(f"human suite coverage not measured: {cov.raw_output[:300]}")
    if cov.line_covered == 0:
        # A real reference suite never exercises zero lines of its own
        # high-complexity module; this means the suite did not actually run.
        raise RuntimeError(
            "human suite covered nothing of the target; check tests_path and test_deps"
        )
    final_tests = tests_path.read_text(encoding="utf-8") if tests_path.is_file() else ""
    result = PipelineResult(final_tests=final_tests, final_coverage=cov, iterations=[],
                            total_tokens=0, stop_reason="human reference suite")
    return build_record(result, module=target.import_name, model=_HUMAN, condition="A", timestamp=timestamp)


def run_condition_b(
    module: TargetModule | str | Path, *, model: ModelConfig, prompts_dir: str | Path,
    token_budget_per_module: int = 60000, generate_fn=generate, timestamp: str | None = None,
) -> ExperimentRecord:
    """Single-pass generation: the loop capped at one iteration, no refinement."""
    target = TargetModule.coerce(module)
    single = LoopConfig(max_iterations=1, token_budget_per_module=token_budget_per_module, stop_on_no_gain=False)
    result = run_pipeline(target, model=model, config=single, prompts_dir=prompts_dir, generate_fn=generate_fn)
    return build_record(result, module=target.import_name, model=model, condition="B", timestamp=timestamp)


def run_condition_c(
    module: TargetModule | str | Path, *, model: ModelConfig, config: LoopConfig, prompts_dir: str | Path,
    generate_fn=generate, timestamp: str | None = None,
) -> ExperimentRecord:
    """The full iterative pipeline (the treatment)."""
    target = TargetModule.coerce(module)
    result = run_pipeline(target, model=model, config=config, prompts_dir=prompts_dir, generate_fn=generate_fn)
    return build_record(result, module=target.import_name, model=model, condition="C", timestamp=timestamp)


def run_module(
    module: TargetModule | str | Path, *, model: ModelConfig, config: LoopConfig, prompts_dir: str | Path,
    human_tests: str | Path | None = None, generate_fn=generate,
    results_dir: str | Path | None = None, timestamp: str | None = None,
) -> list[ExperimentRecord]:
    """Run every available condition on one module and, if a results dir is given, write the records.

    Records and result files are keyed by the dotted import name, never the bare
    file stem: click and jinja2 both ship a ``parser.py``, and their results
    must not overwrite each other.
    """
    target = TargetModule.coerce(module)
    records: list[ExperimentRecord] = []
    if human_tests is not None:
        records.append(run_condition_a(target, human_tests, timestamp=timestamp))
    records.append(run_condition_b(
        target, model=model, prompts_dir=prompts_dir,
        token_budget_per_module=config.token_budget_per_module,
        generate_fn=generate_fn, timestamp=timestamp,
    ))
    records.append(run_condition_c(
        target, model=model, config=config, prompts_dir=prompts_dir,
        generate_fn=generate_fn, timestamp=timestamp,
    ))

    if results_dir is not None:
        results_dir = Path(results_dir)
        for rec in records:
            write_json(rec, results_dir / f"{target.import_name}_{rec.condition}.json")
            append_csv(rec, results_dir / "summary.csv")
    return records
