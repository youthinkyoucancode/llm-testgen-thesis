"""Why generated suites end up empty: the per-round failure-mode breakdown.

The campaign summary reports a coverage percentage per record, so a record that
retained nothing is indistinguishable from a record whose tests all covered
nothing. Those are different outcomes with different causes, and the difference
carries the argument in Chapter 6: a round noted "did not parse" means the model
never produced valid Python, while "no tests passed" means it produced tests
whose assertions were wrong against the unmodified module. The second is the
oracle failure this thesis predicts from Section 2.1.1; the first is not.

This is descriptive only. It does not touch the pre-registered hypothesis test
in analyze_results.py, and it is deliberately kept in a separate script so that
nothing here can influence the paired comparison.

Usage:
    python experiments/failure_modes.py <results_dir> [--out <dir>]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

# Records are named <module>_<condition>_s<seed>.json; mutation sidecars share
# the stem and must not be counted as generation rounds.
RECORD = re.compile(r"^(?P<module>.+)_(?P<condition>[ABC])_s(?P<seed>\d+)$")

# The note strings loop.py can emit, in the order they are reported.
NOTES = ["did not parse", "no tests passed", "no coverage gain"]
KEPT = "tests kept"


def load_records(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        if path.stem.endswith("_mutation"):
            continue
        match = RECORD.match(path.stem)
        if not match:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_module"] = match["module"]
        data["_condition"] = match["condition"]
        data["_seed"] = int(match["seed"])
        records.append(data)
    return records


def classify(iteration: dict) -> str:
    """Map one round to a failure mode, or to the fact that it kept tests."""
    note = (iteration.get("note") or "").strip()
    if note in NOTES:
        return note
    return KEPT if iteration.get("tests_kept") else note or KEPT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = load_records(args.results_dir)
    if not records:
        print(f"no records found in {args.results_dir}")
        return 1

    out_dir = args.out or args.results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_condition: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_module: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    empty_suites: dict[str, int] = collections.Counter()
    totals: dict[str, int] = collections.Counter()

    for record in records:
        condition = record["_condition"]
        if condition == "A":
            continue  # condition A never calls a model, so it has no rounds
        totals[condition] += 1
        rounds = record.get("per_iteration", [])
        if not any(r.get("tests_kept") for r in rounds):
            empty_suites[condition] += 1
        for iteration in rounds:
            mode = classify(iteration)
            by_condition[condition][mode] += 1
            by_module[(record["_module"], condition)][mode] += 1

    modes = NOTES + [KEPT]

    rows = []
    for condition in sorted(by_condition):
        counter = by_condition[condition]
        total = sum(counter.values())
        rows.append([condition, total]
                    + [counter[m] for m in modes]
                    + [f"{100 * counter[m] / total:.1f}" for m in modes])
    header = (["condition", "rounds"] + [f"n_{m.replace(' ', '_')}" for m in modes]
              + [f"pct_{m.replace(' ', '_')}" for m in modes])
    write_csv(out_dir / "failure_modes_by_condition.csv", header, rows)

    module_rows = []
    for (module, condition) in sorted(by_module):
        counter = by_module[(module, condition)]
        module_rows.append([module, condition, sum(counter.values())]
                           + [counter[m] for m in modes])
    write_csv(out_dir / "failure_modes_by_module.csv",
              ["module", "condition", "rounds"] + [m.replace(" ", "_") for m in modes],
              module_rows)

    print(f"records: {len(records)}")
    for condition in sorted(by_condition):
        counter = by_condition[condition]
        total = sum(counter.values())
        empty, n = empty_suites[condition], totals[condition]
        print(f"\ncondition {condition}: {n} records, {total} rounds, "
              f"{empty}/{n} ({100 * empty / n:.0f}%) retained no tests at all")
        for mode in modes:
            if counter[mode]:
                print(f"    {counter[mode]:4d}  ({100 * counter[mode] / total:5.1f}%)  {mode}")
    print(f"\nwrote {out_dir / 'failure_modes_by_condition.csv'}")
    print(f"wrote {out_dir / 'failure_modes_by_module.csv'}")
    return 0


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
