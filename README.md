# llm-testgen-thesis

An iterative, quality-filtered Large Language Model (LLM) pipeline for generating Python unit
tests, with an empirical evaluation against single-pass generation and human-written test suites.

Bachelor thesis artifact, B.Sc. Computer Science, IU International University of Applied Sciences.
The campaign is complete and the analysis tables in this repository are the ones the thesis is
written from.

## What it does

Given a Python module, the pipeline:

1. Extracts the module's structure with the AST.
2. Prompts an LLM to generate candidate unit tests.
3. Filters the candidates through four signals: parse, execute, coverage gain, mutation kill.
4. Feeds the uncovered lines back to the model for a bounded number of refinement rounds.
5. Outputs a filtered pytest suite and a JSON report.

Research question: does iterative filtering by automated quality signals produce measurably better
test suites than single-pass LLM generation? The existing human-written suite is included as a
reference ceiling, not a target to beat.

## Research design

Three conditions are compared, on 12 pre-registered modules across 5 libraries, at seeds 42, 43
and 44:

- Condition A: the existing human-written test suite (reference ceiling), once per module.
- Condition B: single-pass LLM generation (the primary comparison), once per module per seed.
- Condition C: the full iterative, quality-filtered pipeline, once per module per seed.

B and C are the same code path with the iteration cap changed, so any difference between them is
caused by the loop rather than by an unrelated implementation choice.

Metrics: line coverage, branch coverage, mutation score (mutmut). Analysis: Wilcoxon signed-rank
test, Holm-Bonferroni correction, matched-pairs rank-biserial effect size. Real-bug detection on
BugsInPy was scoped out and is carried as future work.

## What the campaign found

Condition C was at least as good as condition B on every non-tied module, on all three metrics
(rank-biserial +1.000), but the margins are small and none of the tests reaches significance: with
five non-zero differences the smallest attainable two-sided p-value is 0.0625, which the design
disclosed in advance. Iteration cost a median of 2.30 times condition B's tokens and returned a
median of 0.585 percentage points of line coverage and 0.000 of mutation score. Both conditions
stay far below the human suites. The full reading, including the threats to validity, is in the
thesis.

## Setup

Requires Python 3.12. The default model is open-weights and runs locally via Ollama, so no paid
API is needed.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.lock
cp .env.example .env               # only needed if you use a closed model
```

For the default (free, local) model:

```
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5-coder
```

Mutation testing uses mutmut 3.x, which does not run on native Windows. The campaign therefore
runs on Linux; `experiments/colab_experiments.ipynb` is the notebook used for it.

## Usage

The package uses a src layout and is not installed, so runs are prefixed with `PYTHONPATH=src`.

```
# one module through the pipeline
PYTHONPATH=src python -m llmtestgen.cli path/to/module.py --condition C --seed 42

# the full campaign (resumable: finished records are skipped)
PYTHONPATH=src python experiments/run_experiments.py --conditions ABC --seeds 42,43,44

# the pre-registered analysis, which writes experiments/results/analysis/
PYTHONPATH=src python experiments/analyze_results.py
```

## What is in here

- `src/llmtestgen/` the pipeline, one component per file: context, generate, the four filters,
  the loop, the report writer, and the A/B/C harness.
- `prompts/` the versioned prompt templates. The version is logged with every run.
- `config/` the run configuration, and `libraries.yaml` with the target libraries and the
  pre-registered module list, pinned to exact versions.
- `experiments/select_modules.py` the mechanical selection rule, with every candidate's metrics,
  including the excluded ones, in `module_selection.json`.
- `experiments/results/analysis/` the frozen tables the thesis reports, plus the exclusion log.
- `tests/` the unit suite for the pipeline itself.

The per-run records and the retained generated suites are large and are archived separately from
this repository.

## Reproducibility

- Dependencies pinned in `requirements.lock`, target libraries pinned in `config/libraries.yaml`.
- Prompts versioned in `prompts/`.
- Every run logs the model id, parameters, prompt version, seed, token counts and latency.
- Note that seeded runs are not bit-identical across hardware, which is why each condition is run
  at three seeds and reported as a distribution rather than as a single run.

## Author

Made by Abir Ben Said.

## License

MIT
