"""Pre-registered target-module selection for the mutation-testing scope.

Menzel condition 4: the modules that receive mutation testing must be chosen by
a rule fixed BEFORE any experiment runs, so the selection cannot chase good
results. This script IS that rule, applied mechanically to the pinned sdists:

1. Download each library's pinned sdist from PyPI (pins in config/libraries.yaml)
   and extract it. The sdist, not the wheel, because experiments also need the
   human test suites that wheels usually omit.
2. Eligible modules: every ``*.py`` under the package source root, excluding
   ``__init__.py``, ``__main__.py``, ``conftest.py``, and any file or directory
   with ``test`` in its name. Modules with zero total complexity are ineligible:
   pure constants (version files, lookup tables) contain nothing to test or
   mutate.
3. Metric: total cyclomatic complexity of the module, the sum of ``radon cc``
   block scores over its functions and methods (classes excluded from the sum,
   they would double-count their methods). Tie-breaks: higher single-block
   complexity, then more source lines of code.
4. Tractability cap: modules over 1200 SLOC (radon raw) are excluded and
   reported. A single such module yields thousands of mutants, more than a free
   Colab session can execute; the exclusion is part of the pre-registration and
   is disclosed in the thesis.
5. Take the top 3 eligible modules per library (all of them if fewer than 3).

Output: a ranked table per library on stdout (including capped rows, marked),
plus ``experiments/module_selection.json`` with every measured module, so the
selection is fully reproducible and auditable. The chosen paths are then
recorded in ``config/libraries.yaml`` under ``mutation_modules``.

Run from the repo root:  python experiments/select_modules.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARIES_YAML = REPO_ROOT / "config" / "libraries.yaml"
SELECTION_JSON = Path(__file__).resolve().parent / "module_selection.json"

SLOC_CAP = 1200
TOP_N = 3
EXCLUDED_NAMES = {"__init__.py", "__main__.py", "conftest.py", "setup.py"}


def download_sdist(name: str, version: str, dest: Path) -> Path:
    """Fetch the pinned sdist via pip and return the extracted source root."""
    subprocess.run(
        [sys.executable, "-m", "pip", "download", "--no-deps",
         "--no-binary", ":all:", f"{name}=={version}", "-d", str(dest)],
        check=True, capture_output=True, text=True,
    )
    archives = list(dest.glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"no sdist archive downloaded for {name}=={version}")
    with tarfile.open(archives[0]) as tar:
        tar.extractall(dest, filter="data")
    extracted = [p for p in dest.iterdir() if p.is_dir()]
    if not extracted:
        raise FileNotFoundError(f"sdist for {name} extracted to nothing")
    return extracted[0]


def find_package_root(sdist_root: Path) -> Path:
    """Locate the importable package directory (handles flat and src/ layouts)."""
    search_base = sdist_root / "src" if (sdist_root / "src").is_dir() else sdist_root
    candidates = [
        p for p in search_base.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and "test" not in p.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"no package directory with __init__.py under {search_base}")
    # Largest candidate wins if several exist (e.g. a stray helper package).
    return max(candidates, key=lambda p: sum(f.stat().st_size for f in p.rglob("*.py")))


def is_eligible(py_file: Path, package_root: Path) -> bool:
    rel_parts = py_file.relative_to(package_root).parts
    if py_file.name in EXCLUDED_NAMES:
        return False
    return not any("test" in part.lower() for part in rel_parts)


def _leaf_blocks(block) -> list:
    """Functions and methods only; classes expand to their methods recursively."""
    methods = getattr(block, "methods", None)
    if methods is None:
        return [block]
    leaves = list(methods)
    for inner in getattr(block, "inner_classes", []):
        leaves.extend(_leaf_blocks(inner))
    return leaves


def measure(py_file: Path) -> dict:
    """Total/max cyclomatic complexity (functions and methods only) plus SLOC."""
    from radon.complexity import cc_visit
    from radon.raw import analyze

    source = py_file.read_text(encoding="utf-8")
    blocks = [leaf for top in cc_visit(source) for leaf in _leaf_blocks(top)]
    complexities = [b.complexity for b in blocks]
    return {
        "total_cc": sum(complexities),
        "max_cc": max(complexities, default=0),
        "blocks": len(complexities),
        "sloc": analyze(source).sloc,
    }


def select_for_library(name: str, version: str, workdir: Path) -> dict:
    sdist_root = download_sdist(name, version, workdir / name)
    package_root = find_package_root(sdist_root)
    rows = []
    for py_file in sorted(package_root.rglob("*.py")):
        if not is_eligible(py_file, package_root):
            continue
        stats = measure(py_file)
        stats["module"] = py_file.relative_to(sdist_root).as_posix()
        stats["capped"] = stats["sloc"] > SLOC_CAP
        rows.append(stats)
    rows.sort(key=lambda r: (-r["total_cc"], -r["max_cc"], -r["sloc"]))
    chosen = [r["module"] for r in rows if not r["capped"] and r["total_cc"] > 0][:TOP_N]
    return {
        "library": name,
        "version": version,
        "package_root": package_root.relative_to(sdist_root).as_posix(),
        "chosen": chosen,
        "all_modules": rows,
    }


def main() -> None:
    config = yaml.safe_load(LIBRARIES_YAML.read_text(encoding="utf-8"))
    results = []
    with tempfile.TemporaryDirectory(prefix="module-selection-") as tmp:
        for lib in config["libraries"]:
            print(f"\n=== {lib['name']} {lib['version']} ===")
            result = select_for_library(lib["name"], lib["version"], Path(tmp))
            results.append(result)
            for row in result["all_modules"]:
                if row["capped"]:
                    marker = "  CAPPED"
                elif row["total_cc"] == 0:
                    marker = "  ZERO-CC"
                elif row["module"] in result["chosen"]:
                    marker = "  <-- selected"
                else:
                    marker = ""
                print(f"  cc={row['total_cc']:4}  max={row['max_cc']:3}  "
                      f"sloc={row['sloc']:5}  {row['module']}{marker}")

    SELECTION_JSON.write_text(
        json.dumps({
            "rule": (f"top {TOP_N} eligible modules per library by total cyclomatic "
                     f"complexity (radon cc, functions and methods), tie-break max block "
                     f"CC then SLOC; excluded: __init__/__main__/conftest/tests, modules "
                     f"with zero total CC (nothing to mutate), and modules over "
                     f"{SLOC_CAP} SLOC (mutation tractability, disclosed)"),
            "libraries": results,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {SELECTION_JSON.relative_to(REPO_ROOT)}")
    print("\nmutation_modules blocks for config/libraries.yaml:")
    for result in results:
        print(f"\n  {result['library']}:")
        for module in result["chosen"]:
            print(f"    - {module}")


if __name__ == "__main__":
    main()
