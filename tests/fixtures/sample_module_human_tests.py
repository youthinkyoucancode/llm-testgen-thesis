"""A hand-written 'human' test suite for sample_module, used as Condition A
(the reference ceiling) when exercising the experiment harness."""

import pytest

from sample_module import Counter, reduce_fraction, slugify


def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"


def test_slugify_no_lowercase():
    assert slugify("HELLO World", lowercase=False) == "HELLO-World"


def test_slugify_strips_punctuation():
    assert slugify("a!b?c") == "abc"


def test_reduce_fraction():
    assert reduce_fraction(98, 42) == (7, 3)


def test_reduce_fraction_zero_denominator():
    with pytest.raises(ZeroDivisionError):
        reduce_fraction(1, 0)


def test_counter_increment_and_positive():
    counter = Counter(0)
    assert counter.increment() == 1
    assert counter.is_positive is True
    assert Counter(-1).is_positive is False
