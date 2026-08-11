"""Model-agnostic generation: one interface, swappable backends.

``generate(prompt, model)`` sends a :class:`~llmtestgen.prompting.Prompt` to the
configured backend (Ollama locally, or the Anthropic / OpenAI APIs) and returns a
:class:`GenerationResult` carrying the raw text plus the reproducibility metadata
the thesis needs: model id, parameters, token counts, and latency.

The architecture calls for ``generate(prompt, model_cfg) -> str``. We return a
small result object instead of a bare string so the report stage can record
tokens and cost without a second round-trip to the backend. Backend SDKs are
imported lazily, so using Ollama never requires the Anthropic or OpenAI packages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .prompting import Prompt


@dataclass
class ModelConfig:
    """How to reach one model. Loaded from the ``model:`` block of a config file."""

    provider: str                  # "ollama" | "anthropic" | "openai"
    name: str
    temperature: float = 0.2
    max_output_tokens: int = 2048
    seed: int | None = None
    base_url: str | None = None    # for OpenAI-compatible endpoints (Colab / vLLM)
    num_gpu: int | None = None     # Ollama: GPU layers to offload (0 = force CPU-only)
    num_ctx: int | None = None     # Ollama: context window in tokens. MUST be set explicitly.

    @classmethod
    def from_yaml(cls, path: str | Path, *, seed: int | None = None) -> ModelConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        m = data["model"]
        return cls(
            provider=m["provider"],
            name=m["name"],
            temperature=m.get("temperature", 0.2),
            max_output_tokens=m.get("max_output_tokens", 2048),
            seed=seed,
            base_url=m.get("base_url"),
            num_gpu=m.get("num_gpu"),
            num_ctx=m.get("num_ctx"),
        )


@dataclass
class GenerationResult:
    """Generated text plus the metadata we log for every call."""

    text: str
    model: str
    provider: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    params: dict = field(default_factory=dict)
    context_truncated: bool = False   # prompt did not fit the context window

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


# Conservative characters-per-token floor. Measured against qwen2.5-coder on a
# real target (click/parser.py: 21,321 characters evaluated as 4,989 tokens, so
# 4.27 chars/token), a chars/4 rule would flag a HEALTHY run as truncated. Eight
# leaves wide margin for any tokenizer while still catching gross truncation:
# the same prompt cut to Ollama's default window evaluated 2,050 tokens, far
# under the 2,665 this floor requires.
_CHARS_PER_TOKEN_FLOOR = 8


def prompt_was_truncated(
    prompt: Prompt, prompt_tokens: int | None, model: "ModelConfig | None" = None
) -> bool:
    """True when the prompt did not reach the model whole.

    Ollama left-truncates a prompt exceeding ``num_ctx`` instead of failing, so
    an oversized prompt yields a plausible-looking result computed from a
    fragment of the module. Two independent checks, because neither alone is
    sufficient:

    * **Headroom.** If the prompt plus the output cap cannot fit inside the
      configured context window, the run is unsafe whether or not the backend
      admits it. This is exact arithmetic over known quantities, no heuristic.
    * **Gross-truncation floor.** If the backend reports implausibly few tokens
      for the character count, something was dropped. Deliberately loose, so it
      never fires on a healthy run.
    """
    if prompt_tokens is None:
        return False
    chars = len(prompt.system) + len(prompt.user)
    if prompt_tokens < chars // _CHARS_PER_TOKEN_FLOOR:
        return True
    if model is not None and model.num_ctx is not None:
        if prompt_tokens + model.max_output_tokens > model.num_ctx:
            return True
    return False


def generate(prompt: Prompt, model: ModelConfig) -> GenerationResult:
    """Dispatch to the configured backend and return the text plus metadata.

    Raises if the prompt did not fit the context window. A truncated prompt is
    not a degraded measurement, it is a measurement of a different input, so it
    must fail loudly rather than be recorded (the same principle that makes a
    zero-coverage human suite an error in ``harness.run_conditions``).
    """
    start = time.perf_counter()
    if model.provider == "ollama":
        text, prompt_tokens, completion_tokens = _generate_ollama(prompt, model)
    elif model.provider == "anthropic":
        text, prompt_tokens, completion_tokens = _generate_anthropic(prompt, model)
    elif model.provider == "openai":
        text, prompt_tokens, completion_tokens = _generate_openai(prompt, model)
    else:
        raise ValueError(f"unknown provider: {model.provider!r}")
    latency = time.perf_counter() - start

    truncated = prompt_was_truncated(prompt, prompt_tokens, model)
    if truncated:
        raise RuntimeError(
            f"prompt did not reach the model whole: {prompt_tokens} prompt tokens "
            f"for {len(prompt.system) + len(prompt.user)} characters, with "
            f"num_ctx={model.num_ctx!r} and max_output_tokens={model.max_output_tokens}. "
            f"Raise model.num_ctx above the prompt size plus the output cap."
        )

    return GenerationResult(
        text=text,
        model=model.name,
        provider=model.provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_s=latency,
        context_truncated=truncated,
        params={
            "temperature": model.temperature,
            "max_output_tokens": model.max_output_tokens,
            "seed": model.seed,
            "num_ctx": model.num_ctx,
        },
    )


def _generate_ollama(prompt: Prompt, model: ModelConfig):
    import ollama

    options = {"temperature": model.temperature, "num_predict": model.max_output_tokens}
    if model.seed is not None:
        options["seed"] = model.seed
    if model.num_gpu is not None:
        options["num_gpu"] = model.num_gpu
    if model.num_ctx is not None:
        # Without this, Ollama applies its own small default and silently drops
        # whatever does not fit. Measured on ollama 0.32.6 with qwen2.5-coder
        # (2026-08-11): a 21,321-character prompt evaluated 2,050 tokens unset
        # versus 4,989 with num_ctx pinned. Target modules run past 12k prompt
        # tokens, so this must be set explicitly for every real run.
        options["num_ctx"] = model.num_ctx
    response = ollama.chat(
        model=model.name,
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        options=options,
    )
    return (
        response.message.content,
        getattr(response, "prompt_eval_count", None),
        getattr(response, "eval_count", None),
    )


def _generate_anthropic(prompt: Prompt, model: ModelConfig):
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model.name,
        max_tokens=model.max_output_tokens,
        temperature=model.temperature,
        system=prompt.system,
        messages=[{"role": "user", "content": prompt.user}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    return text, message.usage.input_tokens, message.usage.output_tokens


def _generate_openai(prompt: Prompt, model: ModelConfig):
    import openai

    client = openai.OpenAI(base_url=model.base_url)
    completion = client.chat.completions.create(
        model=model.name,
        temperature=model.temperature,
        max_tokens=model.max_output_tokens,
        seed=model.seed,
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
    )
    usage = completion.usage
    return (
        completion.choices[0].message.content or "",
        usage.prompt_tokens if usage else None,
        usage.completion_tokens if usage else None,
    )


if __name__ == "__main__":
    import argparse

    from .context import extract_module_context
    from .prompting import build_generation_prompt

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate pytest tests for a module (single pass).")
    parser.add_argument("module", help="path to the target .py file")
    parser.add_argument("--config", default=str(root / "config" / "default.yaml"))
    parser.add_argument("--template", default=str(root / "prompts" / "v1_generate.md"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    context = extract_module_context(args.module)
    prompt = build_generation_prompt(context, args.template)
    model = ModelConfig.from_yaml(args.config, seed=args.seed)

    print(f"provider : {model.provider} / {model.name}  (temp={model.temperature})")
    print(f"prompt   : {prompt.version}, public API = {len(context.public_functions())} callables")
    print("generating... (the first call loads the model into RAM, so it is slow)\n")

    result = generate(prompt, model)

    print("=" * 70)
    print(result.text)
    print("=" * 70)
    toks = (
        f"{result.prompt_tokens} in -> {result.completion_tokens} out"
        if result.total_tokens is not None
        else "token counts n/a"
    )
    print(f"latency: {result.latency_s:.1f}s | {toks} | model: {result.model}")
