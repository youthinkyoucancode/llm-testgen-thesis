"""Pre-registered statistical analysis for the A/B/C campaign (thesis Ch. 5).

Implements exactly the plan fixed in the methodology before the data freeze
(Wilcoxon signed-rank, Holm-Bonferroni across the metric family, rank-biserial
effect sizes), reading the per-record JSONs as the authoritative source (the
flat summary.csv may contain superseded rows after a record redo and is not
used here).

Analysis rules, fixed before the final data freeze:

- Pairing unit: the module. For conditions B and C every metric is collapsed
  to the MEDIAN across the generation seeds, giving one paired value per
  module per condition. Per-seed values are also exported for the
  distribution tables.
- Metric family for hypothesis tests (B vs C): line_percent, branch_percent,
  mutation_score. Holm-Bonferroni corrects across these three comparisons.
- Empty suites: a record with tests_retained == 0 has no tests, so it
  detects no mutants; its effective mutation score is 0.0. (Without this
  rule, modules where single-pass generation produced nothing would silently
  drop out of the mutation comparison, biasing it in B's favor.)
- Incomplete measurements: a record whose suite is non-empty but whose
  mutation_score is missing is an instrument failure, not a result. The
  module is excluded from the mutation comparison and listed loudly.
- Wilcoxon: two-sided, on the differences C minus B; zero differences are
  dropped (scipy zero_method="wilcox") and the effective n is reported. With
  no nonzero differences the test is undefined and reported as such.
- Rank-biserial (matched pairs): (T+ - T-) / (T+ + T-) over the ranks of the
  nonzero absolute differences; positive values favor condition C.
- Condition A is descriptive only (the reference ceiling), never tested.

Usage:
    PYTHONPATH=src python experiments/analyze_results.py \
        --results-dir /content/drive/MyDrive/thesis_results
Outputs land in <results-dir>/analysis/ as CSV and Markdown tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from scipy.stats import wilcoxon

METRICS = ["line_percent", "branch_percent", "mutation_score"]


def load_records(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("_mutation.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if {"module", "condition"} <= record.keys():
            records.append(record)
    return records


def effective_mutation(record: dict) -> float | None:
    """Mutation score under the pre-registered empty-suite rule.

    Returns the recorded score, 0.0 for an empty suite, and None for the
    instrument-failure case (non-empty suite, no score), which the caller
    must exclude and report.
    """
    score = record.get("mutation_score")
    if score is not None:
        return float(score)
    if record.get("tests_retained", 0) == 0:
        return 0.0
    return None


def metric_value(record: dict, metric: str) -> float | None:
    if metric == "mutation_score":
        return effective_mutation(record)
    value = record.get(metric)
    return None if value is None else float(value)


def collapse(records: list[dict]) -> dict[str, dict]:
    """Group records into {module: {"A": record|None, "B": {seed: record}, "C": {...}}}."""
    modules: dict[str, dict] = {}
    for record in records:
        slot = modules.setdefault(
            record["module"], {"A": None, "B": {}, "C": {}}
        )
        condition = record["condition"]
        if condition == "A":
            slot["A"] = record
        elif condition in ("B", "C"):
            slot[condition][record.get("seed")] = record
    return modules


def seed_median(records: dict[int, dict], metric: str) -> tuple[float | None, list[str]]:
    """Median of a metric across seeds; None plus a note when a value is excluded."""
    values, notes = [], []
    for seed, record in sorted(records.items()):
        value = metric_value(record, metric)
        if value is None:
            notes.append(f"seed {seed}: non-empty suite without mutation score")
        else:
            values.append(value)
    if notes or not values:
        return (None, notes) if notes else (None, ["no records"])
    return statistics.median(values), []


def holm_bonferroni(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjustment; None entries (undefined tests) pass through."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (i, p) in enumerate(sorted(indexed, key=lambda x: x[1])):
        running = max(running, (m - rank) * p)
        adjusted[i] = min(1.0, running)
    return adjusted


def paired_test(pairs: list[tuple[str, float, float]]) -> dict:
    """Wilcoxon signed-rank plus matched-pairs rank-biserial for (B, C) pairs."""
    diffs = [c - b for _, b, c in pairs]
    nonzero = [d for d in diffs if d != 0]
    result = {
        "n_pairs": len(pairs),
        "n_nonzero": len(nonzero),
        "median_B": statistics.median([b for _, b, _ in pairs]) if pairs else None,
        "median_C": statistics.median([c for _, _, c in pairs]) if pairs else None,
        "W": None, "p": None, "rank_biserial": None,
    }
    if not nonzero:
        return result

    stat = wilcoxon([d for d in diffs], zero_method="wilcox", alternative="two-sided")
    result["W"] = float(stat.statistic)
    result["p"] = float(stat.pvalue)

    ranked = sorted(nonzero, key=abs)
    ranks: dict[int, float] = {}
    i = 0
    while i < len(ranked):  # average ranks for tied absolute differences
        j = i
        while j < len(ranked) and abs(ranked[j]) == abs(ranked[i]):
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        i = j
    t_plus = sum(rank for k, rank in ranks.items() if ranked[k] > 0)
    t_minus = sum(rank for k, rank in ranks.items() if ranked[k] < 0)
    result["rank_biserial"] = (t_plus - t_minus) / (t_plus + t_minus)
    return result


def fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_markdown(path: Path, title: str, header: list[str], rows: list[list]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(results_dir: Path, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    modules = collapse(load_records(results_dir))
    exclusions: list[str] = []

    # Per-module table: A descriptive plus B/C seed medians, every metric.
    per_module_rows = []
    pairs_by_metric: dict[str, list[tuple[str, float, float]]] = {m: [] for m in METRICS}
    for name in sorted(modules):
        slot = modules[name]
        row = [name]
        for metric in METRICS:
            a_value = metric_value(slot["A"], metric) if slot["A"] else None
            row.append(fmt(a_value, 2 if metric != "mutation_score" else 3))
        medians: dict[str, dict[str, float | None]] = {}
        for condition in ("B", "C"):
            medians[condition] = {}
            for metric in METRICS:
                value, notes = seed_median(slot[condition], metric)
                medians[condition][metric] = value
                for note in notes:
                    if "no records" not in note:
                        exclusions.append(f"{name} {condition}: {note}")
                row.append(fmt(value, 2 if metric != "mutation_score" else 3))
        for metric in METRICS:
            b, c = medians["B"][metric], medians["C"][metric]
            if b is not None and c is not None:
                pairs_by_metric[metric].append((name, b, c))
        per_module_rows.append(row)

    header = ["module"] + [f"{c}_{m}" for c in ("A", "B", "C") for m in METRICS]
    write_csv(out_dir / "per_module.csv", header, per_module_rows)
    write_markdown(out_dir / "per_module.md", "Per-module results (B/C = median over seeds)",
                   header, per_module_rows)

    # Per-seed distribution export for the thesis' distribution tables.
    seed_rows = []
    for name in sorted(modules):
        for condition in ("B", "C"):
            for seed, record in sorted(modules[name][condition].items()):
                seed_rows.append([
                    name, condition, seed,
                    fmt(metric_value(record, "line_percent"), 2),
                    fmt(metric_value(record, "branch_percent"), 2),
                    fmt(metric_value(record, "mutation_score"), 3),
                    record.get("tests_retained"), record.get("total_tokens"),
                ])
    write_csv(out_dir / "per_seed.csv",
              ["module", "condition", "seed", "line_percent", "branch_percent",
               "mutation_score", "tests_retained", "total_tokens"], seed_rows)

    # The pre-registered hypothesis tests.
    tests = {metric: paired_test(pairs_by_metric[metric]) for metric in METRICS}
    adjusted = holm_bonferroni([tests[m]["p"] for m in METRICS])
    test_rows = []
    for metric, p_holm in zip(METRICS, adjusted):
        t = tests[metric]
        t["p_holm"] = p_holm
        test_rows.append([
            metric, t["n_pairs"], t["n_nonzero"], fmt(t["median_B"]), fmt(t["median_C"]),
            fmt(t["W"], 1), fmt(t["p"], 4), fmt(p_holm, 4), fmt(t["rank_biserial"]),
        ])
    test_header = ["metric", "n_pairs", "n_nonzero", "median_B", "median_C",
                   "W", "p", "p_holm", "rank_biserial"]
    write_csv(out_dir / "paired_tests.csv", test_header, test_rows)
    write_markdown(out_dir / "paired_tests.md",
                   "Wilcoxon signed-rank, C vs B (Holm-Bonferroni over the metric family)",
                   test_header, test_rows)

    if exclusions:
        (out_dir / "exclusions.txt").write_text("\n".join(exclusions) + "\n", encoding="utf-8")

    print(f"modules: {len(modules)}")
    for metric in METRICS:
        t = tests[metric]
        print(f"  {metric}: n={t['n_pairs']} median B {fmt(t['median_B'])} -> C {fmt(t['median_C'])}, "
              f"p={fmt(t['p'], 4)} (Holm {fmt(t['p_holm'], 4)}), r_rb={fmt(t['rank_biserial'])}")
    if exclusions:
        print("EXCLUDED (instrument failures, must be zero at the freeze):")
        for line in exclusions:
            print(f"  {line}")
    print(f"tables written to {out_dir}")
    return {"modules": modules, "tests": tests, "exclusions": exclusions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pre-registered campaign analysis.")
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "experiments" / "results"))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.exit(f"results dir not found: {results_dir}")
    analyze(results_dir, Path(args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    main()
