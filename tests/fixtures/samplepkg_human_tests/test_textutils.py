"""Human-written suite for the fixture package (condition A stand-in)."""

from samplepkg.textutils import titlecase, word_count


def test_titlecase_normalizes_and_capitalizes():
    assert titlecase("  heLLo   WORLD ") == "Hello World"


def test_titlecase_empty_input():
    assert titlecase("   ") == ""


def test_word_count_counts_normalized_words():
    assert word_count("one  two\tthree\n") == 3


def test_word_count_empty_input():
    assert word_count("") == 0
