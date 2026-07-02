"""Tests for the experiment harness (fake model for B/C, real coverage for A)."""

from pathlib import Path

from llmtestgen.generate import GenerationResult, ModelConfig
from llmtestgen.harness.run_conditions import run_condition_a, run_module
from llmtestgen.loop import LoopConfig

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "prompts"
FIXTURE = Path(__file__).parent / "fixtures" / "sample_module.py"
HUMAN = Path(__file__).parent / "fixtures" / "sample_module_human_tests.py"
MODEL = ModelConfig(provider="ollama", name="qwen2.5-coder")
STAMP = "2026-07-01T00:00:00+00:00"


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


def test_condition_a_measures_human_suite():
    rec = run_condition_a(FIXTURE, HUMAN, timestamp=STAMP)
    assert rec.condition == "A"
    assert rec.model_name == "human-written"
    assert rec.line_covered > 0
    assert rec.tests_retained >= 5


def test_run_module_all_conditions(tmp_path):
    cfg = LoopConfig(max_iterations=2, token_budget_per_module=100_000, stop_on_no_gain=False)
    # B consumes 1 response; C consumes 2 (max_iterations=2). Three total.
    records = run_module(FIXTURE, model=MODEL, config=cfg, prompts_dir=PROMPTS,
                         human_tests=HUMAN, generate_fn=scripted(FULLER, PARTIAL, FULLER),
                         results_dir=tmp_path, timestamp=STAMP)
    assert [r.condition for r in records] == ["A", "B", "C"]
    assert (tmp_path / "sample_module_A.json").exists()
    assert (tmp_path / "sample_module_C.json").exists()
    lines = (tmp_path / "summary.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # header + A + B + C
