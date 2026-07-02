# Prompt template: refinement (v1)
#
# Placeholders filled at runtime: {module_path}, {source_code}, {existing_tests},
#   {uncovered_lines}, {surviving_mutants}, {execution_errors}
# Used in feedback rounds. Log "v1_refine" with every run for reproducibility.

## System
You are an expert Python test engineer improving an existing pytest suite.

## User
The current tests for the module below leave gaps. Add or fix tests to close them.

Focus on:
- Covering these uncovered lines and branches: {uncovered_lines}
- Killing these surviving mutants. A good test should fail if the mutated behavior were the real behavior: {surviving_mutants}
- Fixing these execution errors, if any: {execution_errors}

Constraints:
- All tests must pass against the given, unmodified module.
- Keep the tests that already work; add to or repair them.
- Return only the complete, updated test code in a single Python code block.

Module path: {module_path}

Source code:
```python
{source_code}
```

Existing tests:
```python
{existing_tests}
```
