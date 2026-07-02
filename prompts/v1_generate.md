# Prompt template: generation (v1)
#
# Placeholders filled at runtime: {module_path}, {source_code}, {signatures}
# Keep this file versioned. Log "v1_generate" with every run for reproducibility.

## System
You are an expert Python test engineer. You write correct, idiomatic pytest unit tests.

## User
Write pytest unit tests for the following Python module.

Requirements:
- Use pytest.
- Test the public functions and classes, including edge cases and error paths.
- Each test must be self-contained and must pass against the given, unmodified module.
- Do not test private helpers directly.
- Return only the test code, in a single Python code block, with no explanation.

Module path: {module_path}

Source code:
```python
{source_code}
```

Public API to cover:
{signatures}
