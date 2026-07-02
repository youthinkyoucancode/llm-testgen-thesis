"""Filter 1: parse check.

The model returns test code wrapped in a Markdown fence, sometimes with prose
around it. This filter pulls the Python out of the fence and confirms it actually
parses. Non-parsing candidates are dropped (or, inside the refinement loop, fed
back with the error message), so no later stage ever runs on syntactically broken
code. It is pure syntax: import errors and failing assertions are the execute
filter's job, not this one.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ParseResult:
    ok: bool
    code: str              # extracted code, kept even when it fails to parse (for feedback)
    error: str | None = None


def extract_code(text: str) -> str:
    """Return the code inside the Markdown fence(s), or the whole text if unfenced."""
    blocks = _FENCE.findall(text)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks).strip()
    return text.strip()


def parse_check(text: str) -> ParseResult:
    """Extract code from a model response and confirm it is syntactically valid Python."""
    code = extract_code(text)
    try:
        ast.parse(code)
    except SyntaxError as exc:
        where = f"line {exc.lineno}" if exc.lineno else "unknown line"
        return ParseResult(ok=False, code=code, error=f"{exc.msg} ({where})")
    return ParseResult(ok=True, code=code)
