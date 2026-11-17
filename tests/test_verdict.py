from itertools import product

import pytest

from shadow_judge.verdict import (
    ORDER,
    PERMISSIVE,
    Verdict,
    clamp_to_degradation,
    is_degradation,
    parse,
    severity,
)


def test_scale_runs_permissive_to_restrictive():
    assert ORDER == (Verdict.FULL, Verdict.PARTIAL, Verdict.NONE)
    assert severity(Verdict.FULL) < severity(Verdict.PARTIAL) < severity(Verdict.NONE)
    assert PERMISSIVE is Verdict.FULL


def test_clamp_never_promotes_for_any_pair():
    """The invariant the whole library rests on, checked exhaustively."""
    for original, proposed in product(ORDER, repeat=2):
        result = clamp_to_degradation(original, proposed)
        assert severity(result) >= severity(original), (original, proposed, result)


def test_clamp_lets_a_degradation_through():
    assert clamp_to_degradation(Verdict.FULL, Verdict.NONE) is Verdict.NONE
    assert clamp_to_degradation(Verdict.PARTIAL, Verdict.NONE) is Verdict.NONE


def test_clamp_keeps_the_original_when_asked_to_promote():
    assert clamp_to_degradation(Verdict.NONE, Verdict.FULL) is Verdict.NONE
    assert clamp_to_degradation(Verdict.PARTIAL, Verdict.FULL) is Verdict.PARTIAL


def test_clamp_accepts_plain_strings():
    assert clamp_to_degradation("full", "none") is Verdict.NONE


def test_is_degradation():
    assert is_degradation(Verdict.FULL, Verdict.PARTIAL)
    assert not is_degradation(Verdict.PARTIAL, Verdict.FULL)
    assert not is_degradation(Verdict.NONE, Verdict.NONE)


@pytest.mark.parametrize("raw", ["full", " FULL ", "Partial", "none"])
def test_parse_accepts_sloppy_casing_and_space(raw):
    assert parse(raw) is not None


@pytest.mark.parametrize("raw", ["", "maybe", None, 3, {"verdict": "full"}])
def test_parse_refuses_to_guess(raw):
    assert parse(raw) is None


def test_str_is_the_wire_value():
    assert f"{Verdict.NONE}" == "none"
