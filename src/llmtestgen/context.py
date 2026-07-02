"""Context extraction for the test-generation pipeline.

Reads a target Python module and pulls out what an LLM needs to write good unit
tests for it:

* the public functions and class methods,
* their exact signatures, including decorators, type hints and defaults,
* their docstrings,
* the module's top-level imports,
* the full source under test.

This is the *grounding* step of the pipeline. Everything generated downstream is
anchored to what this file extracts, so it is deliberately deterministic and
dependency-free: standard-library ``ast`` only, no model, no network.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FunctionInfo:
    """One testable callable: a top-level function or a class method."""

    name: str            # "slugify", or "Counter.increment" for a method
    signature: str       # rendered def line(s), decorators included
    docstring: str | None
    source: str          # the callable's own source text
    lineno: int          # 1-based line of the ``def`` in the target file
    is_method: bool
    class_name: str | None = None

    @property
    def is_private(self) -> bool:
        """True for a single-underscore name (private by convention). Dunders count as public API."""
        base = self.name.rsplit(".", 1)[-1]
        if base.startswith("__") and base.endswith("__"):
            return False
        return base.startswith("_")


@dataclass
class ModuleContext:
    """Everything the extractor pulls from one target module."""

    module_path: str
    source: str
    module_docstring: str | None
    imports: list[str]
    functions: list[FunctionInfo]

    def public_functions(self) -> list[FunctionInfo]:
        """Callables that form the module's public surface (the usual test targets)."""
        return [f for f in self.functions if not f.is_private]

    def signatures_block(self, *, include_private: bool = False) -> str:
        """Render signatures (+ first docstring paragraph) for the prompt's ``{signatures}`` slot."""
        funcs = self.functions if include_private else self.public_functions()
        chunks: list[str] = []
        for f in funcs:
            if f.docstring:
                summary = f.docstring.strip().split("\n\n", 1)[0].strip()
                chunks.append(f'{f.signature}\n    """{summary}"""')
            else:
                chunks.append(f.signature)
        return "\n\n".join(chunks)


def extract_module_context(path: str | Path) -> ModuleContext:
    """Parse ``path`` and return its :class:`ModuleContext`.

    Extracts top-level functions and the methods of top-level classes. Nested
    functions and nested classes are intentionally skipped: they are rarely
    direct unit-test targets. Raises ``SyntaxError`` if the file does not parse.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    functions: list[FunctionInfo] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_function_info(node, source))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(_function_info(item, source, class_name=node.name))

    return ModuleContext(
        module_path=str(path),
        source=source,
        module_docstring=ast.get_docstring(tree),
        imports=_collect_imports(tree),
        functions=functions,
    )


def _function_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    class_name: str | None = None,
) -> FunctionInfo:
    name = f"{class_name}.{node.name}" if class_name else node.name
    return FunctionInfo(
        name=name,
        signature=_render_signature(node),
        docstring=ast.get_docstring(node),
        source=ast.get_source_segment(source, node) or "",
        lineno=node.lineno,
        is_method=class_name is not None,
        class_name=class_name,
    )


def _render_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Rebuild a readable signature: decorators, def line, arguments, return type."""
    decorators = [f"@{ast.unparse(d)}" for d in node.decorator_list]
    keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    arguments = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    signature = f"{keyword} {node.name}({arguments}){returns}"
    return "\n".join(decorators + [signature])


def _collect_imports(tree: ast.Module) -> list[str]:
    """Module-level import statements, rendered back to source (function-local imports are ignored)."""
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
    return imports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract test-generation context from a Python module.")
    parser.add_argument("module", help="path to a .py file")
    parser.add_argument("--private", action="store_true", help="include private (underscore) callables")
    args = parser.parse_args()

    ctx = extract_module_context(args.module)

    print(f"module   : {ctx.module_path}")
    if ctx.module_docstring:
        print(f"docstring: {ctx.module_docstring.strip().splitlines()[0]}")
    print(f"imports  : {len(ctx.imports)}")
    for imp in ctx.imports:
        print(f"    {imp}")

    shown = ctx.functions if args.private else ctx.public_functions()
    print(f"callables: {len(shown)} shown ({len(ctx.functions)} total)")
    for f in shown:
        kind = "method" if f.is_method else "function"
        doc = "doc" if f.docstring else "no-doc"
        print(f"    [{kind:8}] {f.name:24} ({doc})")

    print("\n--- signatures block (fills the prompt's {signatures} slot) ---")
    print(ctx.signatures_block(include_private=args.private))
