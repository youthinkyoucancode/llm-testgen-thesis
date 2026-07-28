"""Command-line entry point: run the full pipeline on one module and record it.

    python -m llmtestgen.cli path/to/module.py --condition C

Runs the generate-then-refine loop with the real configured model, prints the
per-round log, and writes the experiment record (JSON + CSV) under the results
directory from the config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .generate import ModelConfig
from .loop import LoopConfig, run_pipeline
from .report import append_csv, build_record, write_json

_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM test-generation pipeline on one module.")
    parser.add_argument("module", help="path to the target .py file")
    parser.add_argument("--condition", default="C", help="experiment condition label (A/B/C)")
    parser.add_argument("--config", default=str(_ROOT / "config" / "default.yaml"))
    parser.add_argument("--prompts-dir", default=str(_ROOT / "prompts"))
    parser.add_argument("--results-dir", default=str(_ROOT / "experiments" / "results"))
    parser.add_argument("--max-iterations", type=int, default=None, help="override the config's loop cap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = ModelConfig.from_yaml(args.config, seed=args.seed)
    loop_cfg = LoopConfig.from_yaml(args.config)
    if args.max_iterations is not None:
        loop_cfg.max_iterations = args.max_iterations

    module_name = Path(args.module).stem
    print(f"module    : {args.module}")
    print(f"model     : {model.provider}/{model.name} (temp={model.temperature}, seed={model.seed})")
    print(f"loop      : max_iterations={loop_cfg.max_iterations}, "
          f"budget={loop_cfg.token_budget_per_module}, stop_on_no_gain={loop_cfg.stop_on_no_gain}")
    print("running... (calling the real model; each round takes a while)\n")

    result = run_pipeline(args.module, model=model, config=loop_cfg, prompts_dir=args.prompts_dir)

    print("round-by-round:")
    for r in result.iterations:
        gain = "gain" if r.gain else "no gain"
        print(f"  [{r.index}] {r.kind:8} kept={r.tests_kept:2} "
              f"lines={r.line_covered:3}/{r.line_percent:5.1f}% branches={r.branch_covered:2} "
              f"tokens={r.tokens:4} -> {gain}  {r.note}")

    cov = result.final_coverage
    print(f"\nstop reason : {result.stop_reason}")
    print(f"total tokens: {result.total_tokens}")
    if cov:
        print(f"final cov   : lines {cov.line_covered}/{cov.line_total} ({cov.line_percent:.1f}%), "
              f"branches {cov.branch_covered}/{cov.branch_total} ({cov.branch_percent:.1f}%)")

    record = build_record(result, module=module_name, model=model, condition=args.condition)
    json_path = write_json(record, Path(args.results_dir) / f"{module_name}_{args.condition}.json")
    csv_path = append_csv(record, Path(args.results_dir) / "summary.csv")
    print(f"\nsaved: {json_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
