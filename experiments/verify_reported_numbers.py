"""Recompute every headline number of the thesis from the committed tables.

The point of this script is independent checkability. The per-run records are
archived outside the repository because of their size, but the aggregate tables
they were reduced to are versioned here, and every figure the thesis reports in
Chapters 5 and 6 follows from those tables by arithmetic that anyone can rerun.

This recomputes the statistics from `per_module.csv` and `per_seed.csv` rather
than reading them out of `paired_tests.csv`, and then compares the results with
the values printed in the thesis. A reader who wants to know whether the write-up
matches its own data can answer that question with one command and no model, no
GPU and no Google account:

    PYTHONPATH=src python experiments/verify_reported_numbers.py

Exit status is 0 when every reported value is reproduced and 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ANALYSIS = Path(__file__).resolve().parent / "results" / "analysis"

# What the thesis says, section by section, so a mismatch names its own claim.
REPORTED = [
    ("5.4  median line coverage, condition B",        "median_B_line",     29.310, 0.001),
    ("5.4  median line coverage, condition C",        "median_C_line",     29.895, 0.001),
    ("5.7  non-zero pairs, line coverage",            "n_nonzero_line",     5,     0),
    ("5.7  Wilcoxon p, line coverage",                "p_line",             0.0625, 0.0001),
    ("5.7  Holm-adjusted p, line coverage",           "p_holm_line",        0.1875, 0.0001),
    ("5.7  rank-biserial, line coverage",             "rb_line",            1.000, 0.001),
    ("5.3  condition-B runs retaining no test",       "empty_B",           21,     0),
    ("5.3  condition-C runs retaining no test",       "empty_C",           20,     0),
    ("5.6  C/B ratio of median tokens per run",       "token_ratio",        2.30,  0.005),
    ("5.6  median of the per-module multiples",       "token_ratio_module", 2.26,  0.005),
    ("5.8  median line-coverage gap, C to A",         "gap_to_human",      68.85,  0.005),
]


def rank_biserial(before: pd.Series, after: pd.Series) -> float:
    """Matched-pairs rank-biserial: the signed share of the rank mass."""
    diff = (after - before)
    diff = diff[diff != 0]
    if diff.empty:
        return float("nan")
    ranks = diff.abs().rank()
    total = ranks.sum()
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / total)


def compute() -> dict[str, float]:
    per_module = pd.read_csv(ANALYSIS / "per_module.csv")
    per_seed = pd.read_csv(ANALYSIS / "per_seed.csv")
    out: dict[str, float] = {}

    # The three pre-registered comparisons, B against C, paired by module.
    p_values, metrics = [], ("line_percent", "branch_percent", "mutation_score")
    for metric in metrics:
        pair = per_module[[f"B_{metric}", f"C_{metric}"]].apply(
            pd.to_numeric, errors="coerce").dropna()
        stat, p = wilcoxon(pair[f"B_{metric}"], pair[f"C_{metric}"],
                           zero_method="wilcox")
        p_values.append(p)
        if metric == "line_percent":
            out["median_B_line"] = pair[f"B_{metric}"].median()
            out["median_C_line"] = pair[f"C_{metric}"].median()
            out["n_nonzero_line"] = int(
                (pair[f"C_{metric}"] != pair[f"B_{metric}"]).sum())
            out["p_line"] = p
            out["rb_line"] = rank_biserial(pair[f"B_{metric}"], pair[f"C_{metric}"])
    out["p_holm_line"] = multipletests(p_values, method="holm")[1][0]

    # Empty suites, counted over every run rather than over the module medians.
    for condition in ("B", "C"):
        runs = per_seed[per_seed["condition"] == condition]
        out[f"empty_{condition}"] = int((runs["tests_retained"] == 0).sum())

    # Cost, reported two ways because they answer different questions. The
    # ratio of the medians describes a typical run; the median of the per-module
    # ratios describes a typical module, and does not let a large module weigh
    # more heavily than a small one. Chapter 5 gives both.
    out["token_ratio"] = (per_seed[per_seed["condition"] == "C"]["total_tokens"].median()
                          / per_seed[per_seed["condition"] == "B"]["total_tokens"].median())
    per_mod = (per_seed.groupby(["module", "condition"])["total_tokens"]
               .median().unstack())
    out["token_ratio_module"] = (per_mod["C"] / per_mod["B"]).median()

    # Distance from the human ceiling, on the modules where A was measurable.
    ceiling = per_module[["A_line_percent", "C_line_percent"]].apply(
        pd.to_numeric, errors="coerce").dropna()
    out["gap_to_human"] = (ceiling["A_line_percent"]
                           - ceiling["C_line_percent"]).median()
    return out


def main() -> int:
    if not (ANALYSIS / "per_module.csv").exists():
        print(f"no analysis tables under {ANALYSIS}", file=sys.stderr)
        return 1

    values = compute()
    width = max(len(label) for label, *_ in REPORTED)
    failures = 0
    print(f"recomputed from {ANALYSIS.relative_to(ANALYSIS.parents[2])}\n")
    for label, key, reported, tolerance in REPORTED:
        got = values[key]
        ok = abs(got - reported) <= tolerance
        failures += not ok
        shown = f"{got:.3f}" if isinstance(got, float) else str(got)
        print(f"  {'ok ' if ok else 'FAIL'}  {label:<{width}}  "
              f"thesis {reported:<8} recomputed {shown}")

    print()
    if failures:
        print(f"{failures} reported value(s) could not be reproduced.")
        return 1
    print(f"all {len(REPORTED)} reported values reproduced from the committed tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
