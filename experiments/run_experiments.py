"""Phase 2 campaign: conditions A/B/C over the pre-registered target modules.

Reads the pinned libraries and module selection from ``config/libraries.yaml``
and runs, per module: condition A once (the library's human suite, no model),
and conditions B and C once per generation seed. Every record is written the
moment it exists and existing records are skipped on re-run, so an interrupted
campaign (Colab sessions die) continues where it stopped instead of starting
over. A module that fails is logged to ``failures.log`` and the campaign moves
on; nothing takes the whole run down.

Isolation note for the methods chapter: target code is imported via
``PYTHONPATH`` precedence in each measurement subprocess, so the pinned sdist
shadows anything in the pipeline's own environment (the pipeline itself depends
on click, and the sdist copy must win). Mutation workspaces get a full copy of
the target package, so mutants never touch the pristine tree.

Usage (from the repo root, typically on Colab):
    PYTHONPATH=src python experiments/run_experiments.py --config config/colab.yaml
Useful flags:
    --only TEXT         run only libraries/modules whose name contains TEXT
    --conditions AB     any subset of ABC (default ABC)
    --seeds 42,43,44    generation seeds; B and C run once per seed
    --skip-mutation     record coverage only (fast first pass)
    --mutation-timeout  seconds per mutmut run (default 1800)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _sdists import fetch_human_tests, fetch_sdist, import_root_for  # noqa: E402
from llmtestgen.generate import ModelConfig  # noqa: E402
from llmtestgen.harness.run_conditions import run_condition_a  # noqa: E402
from llmtestgen.loop import LoopConfig, run_pipeline  # noqa: E402
from llmtestgen.report import append_csv, build_record, write_json  # noqa: E402
from llmtestgen.target import TargetModule  # noqa: E402


def pip_install(packages: list[str]) -> None:
    if packages:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *packages], check=True
        )


def needs_mutation_redo(record: dict, *, skip_mutation: bool) -> bool:
    """True when a banked record is incomplete and must be rebuilt.

    A suite that covered something but has no mutation score means the mutmut
    stage silently measured nothing (every human suite hit this on 2026-08-04:
    workspace config made pytest collect zero tests). Records with an empty
    suite (0% coverage) legitimately carry no mutation score and stay skipped.
    """
    if skip_mutation:
        return False
    return record.get("mutation_score") is None and record.get("line_percent", 0) > 0


def score_mutation(tests, target: TargetModule, *, timeout: float,
                   results_dir: Path, stem: str) -> float | None:
    """Run mutmut and persist the full counts next to the record."""
    from llmtestgen.filters.mutation import run_mutation

    outcome = run_mutation(tests, target, timeout=timeout)
    detail = {
        "killed": outcome.killed, "no_tests": outcome.no_tests,
        "timeout": outcome.timeout, "suspicious": outcome.suspicious,
        "survived": outcome.survived, "skipped": outcome.skipped,
        "total": outcome.total, "score": outcome.score,
        "survivors": outcome.survivors,
    }
    if outcome.total == 0:  # parser found no stats line; keep evidence
        detail["raw_output_tail"] = outcome.raw_output.splitlines()[-15:]
    (results_dir / f"{stem}_mutation.json").write_text(
        json.dumps(detail, indent=2) + "\n", encoding="utf-8"
    )
    return outcome.score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 2 experiment campaign.")
    parser.add_argument("--config", default=str(REPO / "config" / "default.yaml"))
    parser.add_argument("--libraries", default=str(REPO / "config" / "libraries.yaml"))
    parser.add_argument("--results-dir", default=str(REPO / "experiments" / "results"))
    parser.add_argument("--workdir", default=str(REPO / "experiments" / "targets"))
    parser.add_argument("--only", default="", help="substring filter on library or module name")
    parser.add_argument("--conditions", default="ABC")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--skip-mutation", action="store_true")
    parser.add_argument("--mutation-timeout", type=float, default=1800.0)
    parser.add_argument("--coverage-timeout", type=float, default=1800.0,
                        help="seconds for one human-suite coverage run "
                        "(click's suite alone collects 32k+ tests)")
    args = parser.parse_args()

    conditions = args.conditions.upper()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)
    loop_cfg = LoopConfig.from_yaml(args.config)
    libraries = yaml.safe_load(Path(args.libraries).read_text(encoding="utf-8"))["libraries"]

    done, skipped, failures = 0, 0, []

    def attempt(stem: str, work) -> None:
        nonlocal done, skipped
        existing = results_dir / f"{stem}.json"
        if existing.exists():
            try:
                banked = json.loads(existing.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                banked = {}
            if not needs_mutation_redo(banked, skip_mutation=args.skip_mutation):
                skipped += 1
                return
            # The rebuilt record replaces the JSON; summary.csv keeps the
            # superseded row, so the analysis reads the JSONs (or dedupes on
            # the stem keeping the last row).
            print(f"  redo {stem}: non-empty suite without a mutation score", flush=True)
        try:
            record = work()
            write_json(record, results_dir / f"{stem}.json")
            append_csv(record, results_dir / "summary.csv")
            done += 1
            score = "n/a" if record.mutation_score is None else f"{record.mutation_score:.3f}"
            print(f"  ok  {stem}: line {record.line_percent:.1f}%, mutation {score}", flush=True)
        except Exception:
            failures.append(stem)
            (results_dir / "failures.log").open("a", encoding="utf-8").write(
                f"\n=== {stem} ===\n{traceback.format_exc()}\n"
            )
            print(f"  FAIL {stem} (logged to failures.log)", flush=True)

    for lib in libraries:
        modules = lib.get("mutation_modules", [])
        if args.only and args.only not in lib["name"] and not any(args.only in m for m in modules):
            continue
        print(f"\n=== {lib['name']} {lib['version']} ===", flush=True)
        sdist = fetch_sdist(lib["name"], lib["version"], workdir / lib["name"])
        pip_install(lib.get("runtime_deps", []))
        human_tests = None
        if "A" in conditions:
            pip_install(lib.get("test_deps", []))
            try:
                human_tests, provenance = fetch_human_tests(
                    lib["repo"], lib["version"], lib["tests_path"],
                    sdist, workdir / lib["name"],
                )
                print(f"  human tests: {provenance} ({lib['tests_path']})", flush=True)
            except FileNotFoundError as error:
                print(f"  WARN no human tests, condition A skipped: {error}", flush=True)

        for module_rel in modules:
            target = TargetModule.from_path(
                sdist / module_rel, import_root_for(sdist, module_rel)
            )
            name = target.import_name
            if args.only and args.only not in lib["name"] and args.only not in name:
                continue

            if "A" in conditions and human_tests is not None:
                def condition_a():
                    record = run_condition_a(target, human_tests,
                                             coverage_timeout=args.coverage_timeout)
                    if not args.skip_mutation:
                        record.mutation_score = score_mutation(
                            human_tests, target, timeout=args.mutation_timeout,
                            results_dir=results_dir, stem=f"{name}_A",
                        )
                    return record
                attempt(f"{name}_A", condition_a)

            for seed in seeds:
                for cond in (c for c in "BC" if c in conditions):
                    def condition_bc(cond=cond, seed=seed):
                        model = ModelConfig.from_yaml(args.config, seed=seed)
                        cfg = (
                            LoopConfig(max_iterations=1,
                                       token_budget_per_module=loop_cfg.token_budget_per_module,
                                       stop_on_no_gain=False)
                            if cond == "B" else loop_cfg
                        )
                        result = run_pipeline(target, model=model, config=cfg,
                                              prompts_dir=REPO / "prompts")
                        record = build_record(result, module=name, model=model, condition=cond)
                        if not args.skip_mutation and result.final_tests.strip():
                            record.mutation_score = score_mutation(
                                result.final_tests, target, timeout=args.mutation_timeout,
                                results_dir=results_dir, stem=f"{name}_{cond}_s{seed}",
                            )
                        return record
                    attempt(f"{name}_{cond}_s{seed}", condition_bc)

    print(f"\ncampaign: {done} new records, {skipped} already done, {len(failures)} failures")
    if failures:
        print("failed:", ", ".join(failures))
        print("details in", results_dir / "failures.log")


if __name__ == "__main__":
    main()
