"""Target-module addressing: single files and package modules through one type.

The pipeline grew up on a self-contained fixture file: put its directory on
``PYTHONPATH``, ``import sample_module``, done. The real experiment targets live
inside packages (``src/click/types.py``) and use relative imports, so three
facts must travel together: which FILE is under test, which DIRECTORY makes it
importable, and under what DOTTED NAME tests import it. ``TargetModule``
carries all three. Every filter accepts either a plain path (single-file
semantics, backward compatible) or a ``TargetModule``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetModule:
    path: Path          # the module file under test
    import_root: Path   # the directory placed on sys.path
    import_name: str    # dotted name tests import, e.g. "click.types"

    @classmethod
    def from_path(
        cls, module: str | Path, import_root: str | Path | None = None
    ) -> "TargetModule":
        """Build a target from a module path.

        Without ``import_root`` the module is treated as self-contained: its
        parent directory goes on ``sys.path`` and it imports by file stem. With
        ``import_root`` the dotted name is derived from the path below the root
        (``src`` + ``src/click/types.py`` gives ``click.types``); a module
        outside the root raises ``ValueError`` immediately rather than failing
        later inside a subprocess.
        """
        path = Path(module).resolve()
        if import_root is None:
            return cls(path=path, import_root=path.parent, import_name=path.stem)
        root = Path(import_root).resolve()
        relative = path.relative_to(root)  # ValueError if module is not under root
        return cls(
            path=path,
            import_root=root,
            import_name=".".join(relative.with_suffix("").parts),
        )

    @classmethod
    def coerce(cls, target: "TargetModule | str | Path") -> "TargetModule":
        """Accept what callers already pass: a TargetModule, or a bare path."""
        if isinstance(target, TargetModule):
            return target
        return cls.from_path(target)

    @property
    def top_package(self) -> str | None:
        """The top package directory name, or None for a single-file target."""
        first, _, rest = self.import_name.partition(".")
        return first if rest else None

    @property
    def relative_file(self) -> str:
        """The module file relative to ``import_root``, posix separators."""
        return self.path.relative_to(self.import_root).as_posix()
