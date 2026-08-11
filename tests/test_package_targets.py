"""Package-aware targeting: TargetModule plus every filter on a package fixture.

The fixture package (``tests/fixtures/samplepkg``) contains a relative import,
exactly the property that breaks single-file isolation on real targets like
``click/types.py``. Anything that passes here proves the plumbing the Phase 2
experiments stand on.
"""

import json

from pathlib import Path

import pytest

from llmtestgen.context import extract_module_context
from llmtestgen.filters.coverage_gain import _parse_coverage_json, measure_coverage
from llmtestgen.filters.execute import execute_tests
from llmtestgen.filters.mutation import _prepare_workspace
from llmtestgen.generate import GenerationResult, ModelConfig
from llmtestgen.harness.run_conditions import run_condition_a, run_module
from llmtestgen.loop import LoopConfig, run_pipeline
from llmtestgen.prompting import build_generation_prompt, build_refinement_prompt
from llmtestgen.target import TargetModule

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "prompts"
FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_FILE = FIXTURES / "sample_module.py"
PKG_MODULE = FIXTURES / "samplepkg" / "textutils.py"
HUMAN_DIR = FIXTURES / "samplepkg_human_tests"

MODEL = ModelConfig(provider="fake", name="fake", temperature=0.0, seed=42)

GOOD_SUITE = '''\
from samplepkg.textutils import titlecase, word_count


def test_titlecase_normalizes():
    assert titlecase("  heLLo   WORLD ") == "Hello World"


def test_titlecase_empty():
    assert titlecase("   ") == ""


def test_word_count():
    assert word_count("a  b\\tc") == 3


def test_word_count_empty():
    assert word_count("") == 0
'''


def scripted(*responses: str):
    it = iter(responses)

    def fake(prompt, model):
        return GenerationResult(text=next(it), model="fake", provider="fake",
                                prompt_tokens=100, completion_tokens=100, latency_s=0.0)

    return fake


def pkg_target() -> TargetModule:
    return TargetModule.from_path(PKG_MODULE, import_root=FIXTURES)


# --- TargetModule itself ---------------------------------------------------

def test_single_file_target_keeps_the_old_semantics():
    target = TargetModule.from_path(SINGLE_FILE)
    assert target.import_name == "sample_module"
    assert target.import_root == SINGLE_FILE.resolve().parent
    assert target.top_package is None
    assert target.relative_file == "sample_module.py"


def test_package_target_derives_dotted_name():
    target = pkg_target()
    assert target.import_name == "samplepkg.textutils"
    assert target.top_package == "samplepkg"
    assert target.relative_file == "samplepkg/textutils.py"


def test_module_outside_import_root_fails_loudly():
    with pytest.raises(ValueError):
        TargetModule.from_path(PKG_MODULE, import_root=HUMAN_DIR)


def test_coerce_passes_targets_through_and_wraps_paths():
    target = pkg_target()
    assert TargetModule.coerce(target) is target
    assert TargetModule.coerce(str(SINGLE_FILE)).import_name == "sample_module"


# --- execute + coverage on a package module --------------------------------

def test_execute_resolves_package_relative_imports():
    result = execute_tests(GOOD_SUITE, pkg_target())
    assert result.collected
    assert result.all_passed
    assert len(result.passed) == 4


def test_coverage_measures_the_package_module():
    cov = measure_coverage(GOOD_SUITE, pkg_target())
    assert cov.measured
    assert cov.line_total > 0
    assert cov.line_percent == 100.0


def test_coverage_accepts_a_human_tests_directory():
    cov = measure_coverage(HUMAN_DIR, pkg_target())
    assert cov.measured
    assert cov.line_percent == 100.0


def test_coverage_report_matching_is_suffix_safe(tmp_path):
    # test_x.py must not shadow x.py, and stems alone must never decide.
    report = {
        "files": {
            "proj/test_x.py": {"summary": {"covered_lines": 1, "num_statements": 1,
                                           "covered_branches": 0, "num_branches": 0}},
            "proj/x.py": {"summary": {"covered_lines": 5, "num_statements": 9,
                                      "covered_branches": 1, "num_branches": 2}},
        }
    }
    json_file = tmp_path / "coverage.json"
    json_file.write_text(json.dumps(report), encoding="utf-8")
    parsed = _parse_coverage_json(json_file, "x.py")
    assert parsed.measured
    assert parsed.line_covered == 5 and parsed.line_total == 9


def test_one_uncollectable_module_does_not_void_the_whole_suite(tmp_path):
    """A suite with an uncollectable module must still measure the tests that do run.

    Regression cover for the condition-A defect found on 2026-08-11: markdown
    ships a helper class named ``TestSuite`` that carries an ``__init__``, which
    pytest reports as a collection error. That single error aborted the run
    before any test executed, so coverage recorded only import-time statements.
    markdown's own suite scored 21-28% on modules it actually covers to ~98%,
    and md_in_html scored 0%, which the zero-coverage guardrail caught but the
    other two silently passed.
    """
    suite = tmp_path / "tests"
    suite.mkdir()
    (suite / "test_uncollectable.py").write_text(
        "class TestSuite:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n",
        encoding="utf-8",
    )
    (suite / "test_real.py").write_text(GOOD_SUITE, encoding="utf-8")

    cov = measure_coverage(suite, pkg_target(), timeout=300)

    assert cov.measured, cov.raw_output
    assert cov.line_covered > 0, (
        "the collectable tests did not run; one uncollectable module aborted the "
        f"whole suite again. pytest said:\n{cov.raw_output}"
    )


# --- mutation workspace layout (pure file ops, runs on any OS) -------------

def test_mutation_workspace_carries_the_whole_package(tmp_path):
    mutate_path = _prepare_workspace(GOOD_SUITE, pkg_target(), tmp_path)
    assert mutate_path == "samplepkg/textutils.py"
    assert (tmp_path / "samplepkg" / "textutils.py").exists()
    assert (tmp_path / "samplepkg" / "_helpers.py").exists()      # the relative import target
    assert (tmp_path / "tests" / "test_generated.py").read_text(encoding="utf-8") == GOOD_SUITE
    pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    # mutmut builds its mutants tree from source_paths alone, so the whole
    # package must be in there (sibling imports) with mutation narrowed to
    # the target module; learned from more-itertools' human suite failing
    # with "module 'more_itertools' has no attribute 'peekable'" on Colab.
    assert 'source_paths = ["samplepkg"]' in pyproject
    assert 'only_mutate = ["samplepkg/textutils.py"]' in pyproject
    # The collection flag rides along here: one uncollectable module in a real
    # library's suite otherwise aborts pytest before any test runs, which voids
    # the mutation score instead of lowering it (markdown, condition A).
    assert ('pytest_add_cli_args_test_selection = '
            '["tests/", "--continue-on-collection-errors"]') in pyproject
    assert "paths_to_mutate" not in pyproject
    assert "tests_dir" not in pyproject


def test_mutation_workspace_config_matches_installed_mutmut(tmp_path, monkeypatch, recwarn):
    """The installed mutmut must interpret our workspace config as intended."""
    mutmut_config = pytest.importorskip("mutmut.configuration")
    _prepare_workspace(GOOD_SUITE, pkg_target(), tmp_path)
    monkeypatch.chdir(tmp_path)
    mutmut_config.Config.reset()
    try:
        cfg = mutmut_config.Config.get()
        assert [str(p) for p in cfg.source_paths] == ["samplepkg"]
        assert cfg.pytest_add_cli_args_test_selection == [
            "tests/", "--continue-on-collection-errors"
        ]
        assert cfg.should_mutate("samplepkg/textutils.py")
        assert not cfg.should_mutate("samplepkg/_helpers.py")
        deprecations = [w for w in recwarn if "deprecated" in str(w.message).lower()]
        assert not deprecations
    finally:
        mutmut_config.Config.reset()


def test_mutation_workspace_renames_undiscoverable_suite_files(tmp_path):
    # python-slugify ships its suite as "test.py", which pytest's default
    # discovery never collects; staged under a compliant name instead.
    suite = tmp_path / "src_suite"
    suite.mkdir()
    human = suite / "test.py"
    human.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _prepare_workspace(human, pkg_target(), workspace)
    assert (workspace / "tests" / "test_human_suite.py").exists()
    assert not (workspace / "tests" / "test.py").exists()


def test_mutation_workspace_accepts_a_tests_directory(tmp_path):
    _prepare_workspace(HUMAN_DIR, pkg_target(), tmp_path)
    assert (tmp_path / "tests" / "test_textutils.py").exists()
    assert (tmp_path / "tests" / "conftest.py").exists()          # suite plumbing travels along


def test_mutation_workspace_single_file_layout_unchanged(tmp_path):
    mutate_path = _prepare_workspace("def test_ok():\n    assert True\n",
                                     TargetModule.from_path(SINGLE_FILE), tmp_path)
    assert mutate_path == "sample_module.py"
    assert (tmp_path / "sample_module.py").exists()
    pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'source_paths = ["sample_module.py"]' in pyproject
    assert "only_mutate" not in pyproject


# --- prompts name the import explicitly ------------------------------------

def test_generation_prompt_carries_the_dotted_import_name():
    context = extract_module_context(PKG_MODULE)
    prompt = build_generation_prompt(context, PROMPTS / "v1_generate.md",
                                     import_name="samplepkg.textutils")
    assert "`samplepkg.textutils`" in prompt.user
    assert "{import_name}" not in prompt.user


def test_refinement_prompt_carries_the_dotted_import_name():
    context = extract_module_context(PKG_MODULE)
    prompt = build_refinement_prompt(context, PROMPTS / "v1_refine.md",
                                     existing_tests="# none", uncovered_lines="none",
                                     import_name="samplepkg.textutils")
    assert "`samplepkg.textutils`" in prompt.user
    assert "{import_name}" not in prompt.user


def test_prompt_falls_back_to_the_file_stem():
    context = extract_module_context(SINGLE_FILE)
    prompt = build_generation_prompt(context, PROMPTS / "v1_generate.md")
    assert "`sample_module`" in prompt.user


# --- condition A refuses to record a suite that did not really run ----------

def test_condition_a_rejects_a_missing_suite():
    with pytest.raises(FileNotFoundError):
        run_condition_a(pkg_target(), FIXTURES / "does_not_exist")


def test_condition_a_rejects_a_suite_that_covers_nothing(tmp_path):
    # A suite that runs fine but never imports the target must fail loudly,
    # not enter the results as a 0% "ceiling".
    (tmp_path / "test_unrelated.py").write_text(
        "def test_unrelated():\n    assert True\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="covered nothing"):
        run_condition_a(pkg_target(), tmp_path)


# --- the loop and the harness end to end on the package fixture ------------

def test_pipeline_runs_on_a_package_module():
    cfg = LoopConfig(max_iterations=1, token_budget_per_module=100_000, stop_on_no_gain=False)
    result = run_pipeline(pkg_target(), model=MODEL, config=cfg, prompts_dir=PROMPTS,
                          generate_fn=scripted(GOOD_SUITE))
    assert result.final_coverage is not None
    assert result.final_coverage.line_percent == 100.0
    assert result.iterations[0].tests_kept == 4


def test_run_module_keys_records_by_import_name(tmp_path):
    cfg = LoopConfig(max_iterations=1, token_budget_per_module=100_000, stop_on_no_gain=False)
    records = run_module(pkg_target(), model=MODEL, config=cfg, prompts_dir=PROMPTS,
                         human_tests=HUMAN_DIR, generate_fn=scripted(GOOD_SUITE, GOOD_SUITE),
                         results_dir=tmp_path, timestamp="2026-08-03T00:00:00+00:00")
    assert [r.condition for r in records] == ["A", "B", "C"]
    assert all(r.module == "samplepkg.textutils" for r in records)
    for condition in "ABC":
        assert (tmp_path / f"samplepkg.textutils_{condition}.json").exists()
    assert records[0].line_percent == 100.0      # the human suite is the ceiling here
