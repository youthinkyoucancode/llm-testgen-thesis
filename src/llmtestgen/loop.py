"""The iterative refinement loop (Menzel condition 3).

This ties the pieces into the actual pipeline. Round 0 generates tests from the
module; each later round shows the model the current tests plus what is still
uncovered and asks it to improve them. Every round is filtered (parse, execute,
coverage) and logged. The loop stops on an explicit, logged condition: the
iteration cap, the per-module token budget, or convergence (a round that adds no
new coverage). Those termination rules are what make the comparison fair and the
run reproducible.

The model call is injected via ``generate_fn`` so the orchestration can be tested
with a fake model, without paying for slow real generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .context import extract_module_context
from .filters.coverage_gain import CoverageResult, measure_coverage
from .filters.execute import execute_tests, keep_passing
from .filters.parse_check import parse_check
from .generate import GenerationResult, ModelConfig, generate
from .prompting import build_generation_prompt, build_refinement_prompt


@dataclass
class LoopConfig:
    max_iterations: int = 3
    token_budget_per_module: int = 60000
    stop_on_no_gain: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> LoopConfig:
        loop = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("loop", {})
        return cls(
            max_iterations=loop.get("max_iterations", 3),
            token_budget_per_module=loop.get("token_budget_per_module", 60000),
            stop_on_no_gain=loop.get("stop_on_no_gain", True),
        )


@dataclass
class IterationRecord:
    index: int
    kind: str                 # "generate" or "refine"
    prompt_version: str
    tokens: int
    tests_kept: int
    line_covered: int
    branch_covered: int
    line_percent: float
    gain: bool
    note: str = ""


@dataclass
class PipelineResult:
    final_tests: str
    final_coverage: CoverageResult | None
    iterations: list[IterationRecord] = field(default_factory=list)
    total_tokens: int = 0
    stop_reason: str = ""


def run_pipeline(
    target_module: str | Path,
    *,
    model: ModelConfig,
    config: LoopConfig,
    prompts_dir: str | Path,
    generate_fn=generate,
) -> PipelineResult:
    """Run the generate-then-refine loop on one module; return the best suite plus a per-round log."""
    context = extract_module_context(target_module)
    prompts_dir = Path(prompts_dir)

    best_code = ""
    best_cov: CoverageResult | None = None
    records: list[IterationRecord] = []
    total_tokens = 0
    execution_errors = ""
    stop_reason = ""

    for i in range(config.max_iterations):
        if i == 0:
            prompt = build_generation_prompt(context, prompts_dir / "v1_generate.md")
            kind = "generate"
        else:
            prompt = build_refinement_prompt(
                context,
                prompts_dir / "v1_refine.md",
                existing_tests=best_code or "# no working tests yet",
                uncovered_lines=_format_uncovered(best_cov),
                execution_errors=execution_errors,
            )
            kind = "refine"

        result: GenerationResult = generate_fn(prompt, model)
        total_tokens += result.total_tokens or 0

        prev_score = _score(best_cov)
        note = ""
        cov: CoverageResult | None = None
        kept = ""
        tests_kept = 0

        parsed = parse_check(result.text)
        if not parsed.ok:
            execution_errors = f"syntax error: {parsed.error}"
            note = "did not parse"
        else:
            executed = execute_tests(parsed.code, target_module)
            execution_errors = "\n".join(o.message for o in executed.failed)
            if executed.any_passed:
                kept = keep_passing(parsed.code, executed)
                tests_kept = len(executed.passed)
                cov = measure_coverage(kept, target_module)
            else:
                note = "no tests passed"

        improved = sum(_score(cov)) > sum(prev_score)
        if improved:
            best_code, best_cov = kept, cov

        records.append(IterationRecord(
            index=i, kind=kind, prompt_version=prompt.version,
            tokens=result.total_tokens or 0, tests_kept=tests_kept,
            line_covered=cov.line_covered if cov else 0,
            branch_covered=cov.branch_covered if cov else 0,
            line_percent=cov.line_percent if cov else 0.0,
            gain=improved, note=note,
        ))

        if total_tokens >= config.token_budget_per_module:
            stop_reason = "token budget spent"
            break
        if i > 0 and config.stop_on_no_gain and not improved:
            stop_reason = "no coverage gain"
            break
    else:
        stop_reason = "max iterations reached"

    return PipelineResult(
        final_tests=best_code, final_coverage=best_cov,
        iterations=records, total_tokens=total_tokens, stop_reason=stop_reason,
    )


def _score(cov: CoverageResult | None) -> tuple[int, int]:
    return (cov.line_covered, cov.branch_covered) if cov else (0, 0)


def _format_uncovered(cov: CoverageResult | None) -> str:
    if cov is None:
        return "the tests do not run yet; first make them import and pass against the module"
    parts = []
    if cov.missing_lines:
        parts.append("lines " + ", ".join(map(str, cov.missing_lines)))
    if cov.missing_branches:
        parts.append("branches " + ", ".join(f"{a}->{b}" for a, b in cov.missing_branches))
    return "; ".join(parts) if parts else "none (full coverage reached)"
