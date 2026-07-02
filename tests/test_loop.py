"""Tests for the iterative loop, using a scripted fake model (no real generation)."""

from pathlib import Path

from llmtestgen.generate import GenerationResult, ModelConfig
from llmtestgen.loop import LoopConfig, run_pipeline

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "prompts"
FIXTURE = Path(__file__).parent / "fixtures" / "sample_module.py"

MODEL = ModelConfig(provider="ollama", name="qwen2.5-coder")


def fenced(code: str) -> str:
    return f"```python\n{code}\n```"


PARTIAL = fenced(
    'from sample_module import slugify\n\n\n'
    'def test_slug():\n'
    '    assert slugify("Hello World") == "hello-world"\n'
)

FULLER = fenced(
    'import pytest\n'
    'from sample_module import slugify, reduce_fraction, Counter\n\n\n'
    'def test_slug():\n'
    '    assert slugify("Hello World") == "hello-world"\n\n\n'
    'def test_slug_no_lower():\n'
    '    assert slugify("HELLO", lowercase=False) == "HELLO"\n\n\n'
    'def test_reduce():\n'
    '    assert reduce_fraction(98, 42) == (7, 3)\n\n\n'
    'def test_reduce_zero():\n'
    '    with pytest.raises(ZeroDivisionError):\n'
    '        reduce_fraction(1, 0)\n\n\n'
    'def test_counter():\n'
    '    c = Counter()\n'
    '    assert c.increment(3) == 3\n'
    '    assert c.is_positive is True\n'
)


def scripted(*responses: str):
    it = iter(responses)

    def fake(prompt, model):
        return GenerationResult(text=next(it), model="fake", provider="fake",
                                prompt_tokens=100, completion_tokens=100, latency_s=0.0)

    return fake


def test_stops_on_no_gain():
    cfg = LoopConfig(max_iterations=3, token_budget_per_module=10_000, stop_on_no_gain=True)
    result = run_pipeline(FIXTURE, model=MODEL, config=cfg, prompts_dir=PROMPTS,
                          generate_fn=scripted(PARTIAL, FULLER, FULLER))
    assert result.iterations[0].kind == "generate"
    assert result.iterations[1].kind == "refine"
    assert result.iterations[1].gain is True     # fuller improves on partial
    assert result.iterations[2].gain is False    # same suite again, no new coverage
    assert result.stop_reason == "no coverage gain"
    assert "test_reduce" in result.final_tests    # adopted the fuller suite
    assert result.final_coverage.line_covered > 0


def test_stops_on_token_budget():
    cfg = LoopConfig(max_iterations=5, token_budget_per_module=300, stop_on_no_gain=False)
    result = run_pipeline(FIXTURE, model=MODEL, config=cfg, prompts_dir=PROMPTS,
                          generate_fn=scripted(PARTIAL, FULLER, FULLER, FULLER, FULLER))
    assert result.stop_reason == "token budget spent"
    assert len(result.iterations) == 2            # 200 tokens/round, cap 300 -> stop after round 1


def test_runs_to_max_iterations():
    cfg = LoopConfig(max_iterations=2, token_budget_per_module=100_000, stop_on_no_gain=False)
    result = run_pipeline(FIXTURE, model=MODEL, config=cfg, prompts_dir=PROMPTS,
                          generate_fn=scripted(PARTIAL, PARTIAL))
    assert result.stop_reason == "max iterations reached"
    assert len(result.iterations) == 2
