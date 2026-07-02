# llm-testgen-thesis

An iterative, quality-filtered Large Language Model (LLM) pipeline for generating Python unit tests, with an empirical evaluation against single-pass generation and human-written test suites.

Bachelor thesis artifact, B.Sc. Computer Science, IU International University of Applied Sciences.

## What it does

Given a Python module, the pipeline:

1. Extracts the module's structure with the AST.
2. Prompts an LLM to generate candidate unit tests.
3. Filters the candidates through four signals: parse, execute, coverage gain, mutation kill.
4. Feeds failures back to the model for a bounded number of refinement rounds.
5. Outputs a filtered pytest suite and a JSON report.

Research question: does iterative filtering by automated quality signals produce measurably better test suites than single-pass LLM generation? The existing human-written suite is included as a reference ceiling, not a target to beat.

## Research design

Three conditions are compared per target library:

- Baseline A: the existing human-written test suite (reference ceiling).
- Baseline B: single-pass LLM generation, no filtering (the primary comparison).
- Treatment: the full iterative, quality-filtered pipeline.

Metrics: line coverage, branch coverage, mutation score, real-bug detection (BugsInPy). Analysis: Wilcoxon signed-rank test, Holm-Bonferroni correction, rank-biserial effect size.

## Setup

Requires Python 3.12. The default model is open-weights and runs locally via Ollama, so no paid API is needed.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # only needed if you use a closed model
```

For the default (free, local) model:

```
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5-coder
```

## Usage (planned, build in progress)

```
python -m llmtestgen generate --module path/to/module.py --config config/default.yaml
python -m harness.run_conditions --library click
python -m harness.analysis
```

## Reproducibility

- Dependencies pinned in requirements.txt (freeze to requirements.lock before the first real run).
- Prompts versioned in prompts/.
- Every run logs the model id, prompt version, seed, and token usage.
- Target libraries pinned to exact versions in config/libraries.yaml.

## Author

Made by Abir Ben Said.

## License

MIT
