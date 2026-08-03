"""Shared sdist plumbing for the experiment scripts: download, extract, locate.

Sdists rather than wheels because the experiments also need each library's
human-written test suite, which wheels usually omit. Downloads are cached per
library directory, so interrupted campaigns resume without refetching.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path


def fetch_sdist(name: str, version: str, dest: Path) -> Path:
    """Download and extract the pinned sdist into ``dest``; return the source root."""
    dest.mkdir(parents=True, exist_ok=True)
    existing = [p for p in dest.iterdir() if p.is_dir()]
    if existing:
        return existing[0]
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
    return next(p for p in dest.iterdir() if p.is_dir())


def import_root_for(sdist_root: Path, module_rel: str) -> Path:
    """The directory that must go on sys.path for this module path to import.

    ``src/click/types.py`` needs ``<sdist>/src``; a flat layout like
    ``markdown/inlinepatterns.py`` needs the sdist root itself.
    """
    first = Path(module_rel).parts[0]
    return sdist_root / "src" if first == "src" else sdist_root


def fetch_human_tests(
    repo_url: str, version: str, tests_rel: str, sdist_root: Path, dest: Path
) -> tuple[Path, str]:
    """Locate the human test suite; returns (path, provenance).

    Preferred source is the sdist itself, but not every project ships its tests
    there (python-slugify does not), so the fallback is the project's tagged
    release tarball on GitHub, trying both ``v1.2.3`` and ``1.2.3`` tag styles.
    The provenance string goes into the campaign log so the thesis can state
    where each suite came from.
    """
    shipped = sdist_root / tests_rel
    if shipped.exists():
        return shipped, "sdist"

    for existing in dest.glob("repo-*/"):
        candidate = existing / tests_rel
        if candidate.exists():
            return candidate, f"github tag (cached, {existing.name.removeprefix('repo-')})"

    repo_url = repo_url.rstrip("/")
    for tag in (f"v{version}", version):
        url = f"{repo_url}/archive/refs/tags/{tag}.tar.gz"
        archive = dest / f"repo-{tag}.tar.gz"
        try:
            urllib.request.urlretrieve(url, archive)
        except urllib.error.HTTPError:
            continue
        extract_dir = dest / f"repo-{tag}"
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(extract_dir, filter="data")
        inner = next(p for p in extract_dir.iterdir() if p.is_dir())
        # Flatten one level so the cache glob above finds it next time.
        for item in inner.iterdir():
            item.rename(extract_dir / item.name)
        inner.rmdir()
        candidate = extract_dir / tests_rel
        if candidate.exists():
            return candidate, f"github tag {tag}"
    raise FileNotFoundError(
        f"human tests '{tests_rel}' found neither in the sdist nor in a GitHub "
        f"release tag of {repo_url} for version {version}"
    )
