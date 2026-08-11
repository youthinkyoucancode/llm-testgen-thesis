"""Tests for the generation layer, in particular the context-truncation guard.

Regression cover for the defect that voided the 2026-08-10 campaign: Ollama
applies its own small default context window and silently left-truncates any
prompt that exceeds it, so oversized targets were scored on a fragment of the
module instead of the module. The guard turns that silent corruption into a
loud failure. No real model is called here.
"""

from pathlib import Path

import pytest
import yaml

from llmtestgen.generate import (
    ModelConfig,
    generate,
    prompt_was_truncated,
)
from llmtestgen.prompting import Prompt

BIG = Prompt(version="v1_generate", system="sys", user="x" * 40_000)   # >= 10k tokens
SMALL = Prompt(version="v1_generate", system="sys", user="short prompt")


def test_truncation_detected_when_backend_evaluates_too_few_tokens():
    # 40k chars cannot possibly tokenize to 4098 tokens; part of it was dropped.
    assert prompt_was_truncated(BIG, 4098) is True


def test_no_truncation_when_token_count_clears_the_lower_bound():
    assert prompt_was_truncated(BIG, 12_000) is False


def test_no_truncation_for_a_small_prompt():
    assert prompt_was_truncated(SMALL, 6) is False


def test_unknown_token_count_is_not_reported_as_truncation():
    # Backends that report no usage must not trip the guard.
    assert prompt_was_truncated(BIG, None) is False


def test_healthy_run_at_the_measured_token_ratio_is_not_flagged():
    """Regression: the guard must not fire on a real, complete generation.

    Measured live against qwen2.5-coder on click/parser.py on 2026-08-11: a
    21,321-character prompt evaluated as 4,989 tokens, a ratio of 4.27
    characters per token. An earlier version of this guard used chars/4 as its
    floor and would have aborted EVERY record of the repaired campaign.
    """
    real = Prompt(version="v1_generate", system="s" * 321, user="u" * 21_000)
    model = ModelConfig(provider="ollama", name="qwen2.5-coder",
                        num_ctx=32768, max_output_tokens=2048)
    assert prompt_was_truncated(real, 4989, model) is False


def test_same_prompt_at_the_default_window_is_flagged():
    """The other half of the same measurement: unset num_ctx evaluated 2,050."""
    real = Prompt(version="v1_generate", system="s" * 321, user="u" * 21_000)
    assert prompt_was_truncated(real, 2050) is True


def test_prompt_leaving_no_room_for_the_completion_is_flagged():
    """Fits the character floor, but the output cap cannot fit beside it."""
    model = ModelConfig(provider="ollama", name="qwen2.5-coder",
                        num_ctx=8192, max_output_tokens=2048)
    prompt = Prompt(version="v1_generate", system="s", user="u" * 40_000)
    assert prompt_was_truncated(prompt, 7000, model) is True


def test_generate_raises_instead_of_recording_a_truncated_run(monkeypatch):
    model = ModelConfig(provider="ollama", name="qwen2.5-coder", num_ctx=4096)
    monkeypatch.setattr(
        "llmtestgen.generate._generate_ollama",
        lambda prompt, model: ("```python\n# tests\n```", 4098, 100),
    )
    with pytest.raises(RuntimeError, match="did not reach the model whole"):
        generate(BIG, model)


def test_generate_succeeds_and_records_num_ctx_when_the_prompt_fits(monkeypatch):
    model = ModelConfig(provider="ollama", name="qwen2.5-coder", num_ctx=32768)
    monkeypatch.setattr(
        "llmtestgen.generate._generate_ollama",
        lambda prompt, model: ("```python\n# tests\n```", 12_000, 100),
    )
    result = generate(BIG, model)
    assert result.context_truncated is False
    assert result.params["num_ctx"] == 32768
    assert result.total_tokens == 12_100


def test_num_ctx_is_read_from_the_config_file(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "model": {"provider": "ollama", "name": "qwen2.5-coder", "num_ctx": 32768}
    }), encoding="utf-8")
    assert ModelConfig.from_yaml(cfg).num_ctx == 32768


def test_shipped_configs_set_a_context_window_large_enough_for_the_targets():
    """Both configs must pin num_ctx; the default is what silently corrupted the run."""
    repo = Path(__file__).resolve().parents[1]
    for name in ("default.yaml", "colab.yaml"):
        model = yaml.safe_load((repo / "config" / name).read_text(encoding="utf-8"))["model"]
        assert model.get("num_ctx"), f"{name} does not pin num_ctx"
        # Largest pre-registered target needs ~13k prompt tokens plus the output cap.
        assert model["num_ctx"] >= 13_000 + model.get("max_output_tokens", 2048)
