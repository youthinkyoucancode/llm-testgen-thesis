"""Prompt construction: turn a ModuleContext into the messages sent to the model.

Templates live as versioned markdown files in ``prompts/`` (for example
``v1_generate.md``) with ``## System`` and ``## User`` sections and
``{placeholder}`` slots. Keeping them as data files, separate from code, is what
lets us log a prompt *version* with every run and reproduce it later, which is
one of the reproducibility commitments in the exposé.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .context import ModuleContext


@dataclass
class Prompt:
    """A rendered prompt, ready to send. ``version`` is logged with every run."""

    version: str
    system: str
    user: str


def build_generation_prompt(
    context: ModuleContext,
    template_path: str | Path,
    *,
    import_name: str | None = None,
) -> Prompt:
    """Fill the generation template with the module's source and public signatures.

    ``import_name`` is the dotted name tests must import (``click.types``);
    when omitted it falls back to the file stem, the single-file convention.
    """
    template_path = Path(template_path)
    system, user = _split_sections(template_path.read_text(encoding="utf-8"))
    user = _fill(
        user,
        {
            "module_path": context.module_path,
            "import_name": import_name or Path(context.module_path).stem,
            "source_code": context.source,
            "signatures": context.signatures_block(),
        },
    )
    return Prompt(version=template_path.stem, system=system.strip(), user=user.strip())


def build_refinement_prompt(
    context: ModuleContext,
    template_path: str | Path,
    *,
    existing_tests: str,
    uncovered_lines: str,
    surviving_mutants: str = "",
    execution_errors: str = "",
    import_name: str | None = None,
) -> Prompt:
    """Fill the refinement template with the current tests and this round's feedback."""
    template_path = Path(template_path)
    system, user = _split_sections(template_path.read_text(encoding="utf-8"))
    user = _fill(
        user,
        {
            "module_path": context.module_path,
            "import_name": import_name or Path(context.module_path).stem,
            "source_code": context.source,
            "existing_tests": existing_tests,
            "uncovered_lines": uncovered_lines,
            "surviving_mutants": surviving_mutants or "none reported",
            "execution_errors": execution_errors or "none",
        },
    )
    return Prompt(version=template_path.stem, system=system.strip(), user=user.strip())


def _split_sections(text: str) -> tuple[str, str]:
    """Split a template into (system, user) on its ``## System`` / ``## User`` headers."""
    if "## System" not in text or "## User" not in text:
        raise ValueError("template must contain '## System' and '## User' sections")
    _, after_system = text.split("## System", 1)
    system_part, user_part = after_system.split("## User", 1)
    return system_part, user_part


def _fill(text: str, values: dict[str, str]) -> str:
    """Replace only the known ``{name}`` placeholders.

    Targeted replacement, not ``str.format``, so that literal braces in the
    injected source code (dict/set literals, f-strings) are never mistaken for
    placeholders and never raise a formatting error.
    """
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text
