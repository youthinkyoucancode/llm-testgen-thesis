"""Chapter 5 figures, generated from the banked analysis tables.

Figure 2 shows per-module line coverage with the three seeded runs plotted
individually rather than collapsed to their median, because the between-seed
spread is larger than the between-condition difference and a median-only chart
would hide exactly that.

Usage:
    python experiments/make_figures.py --analysis-dir experiments/results/analysis
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def figure_2(analysis_dir: Path, out_path: Path) -> None:
    rows = list(csv.DictReader((analysis_dir / "per_seed.csv").read_text(encoding="utf-8").splitlines()))
    modules = sorted({r["module"] for r in rows})
    # Longest first would reorder the story; keep the alphabetical order the
    # tables use so a reader can follow one module across chapter and figure.
    style = {"B": ("o", "#4c72b0"), "C": ("^", "#c44e52")}

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for index, module in enumerate(modules):
        for offset, condition in ((-0.13, "B"), (0.13, "C")):
            values = [float(r["line_percent"]) for r in rows
                      if r["module"] == module and r["condition"] == condition]
            marker, colour = style[condition]
            ax.scatter([index + offset] * len(values), values, marker=marker, s=38,
                       facecolors="none", edgecolors=colour, linewidths=1.3, zorder=3,
                       label=f"condition {condition}" if index == 0 else None)

    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels([m.replace("markdown.extensions.", "markdown.") for m in modules],
                       rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Line coverage (%)")
    ax.set_ylim(-4, 100)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    # Upper left: the upper right is where slugify.slugify's 85.87% outlier sits.
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("wrote", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Chapter 5 figures.")
    parser.add_argument("--analysis-dir", default=str(REPO / "experiments" / "results" / "analysis"))
    parser.add_argument("--out-dir", default=str(REPO / "experiments" / "results" / "figures"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_2(Path(args.analysis_dir), out_dir / "fig2_line_coverage_by_seed.png")


if __name__ == "__main__":
    main()
