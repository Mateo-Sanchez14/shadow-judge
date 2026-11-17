"""The verdict scale, and the one rule that makes a judge safe to switch on.

A judge in this library answers one bounded question about an answer a pipeline
has already produced: how much of what was asked does it actually cover? The
scale is ordered from permissive to restrictive:

    FULL  <  PARTIAL  <  NONE

`FULL` is the permissive end on purpose. Every default in this library resolves
toward it, because a judge exists to WITHHOLD answers: an unset or unreadable
verdict must let the turn through unchanged rather than silently swallow it.

The invariant this module exists to enforce is that a judge can only ever
*degrade* a turn. There is no verdict that turns a withheld answer into a sent
one. That is what makes it safe to put in front of live traffic: the worst a
broken judge can do is hold back answers that would have been fine, which is
visible and recoverable. A judge that could also promote would be able to send
an answer nobody's pipeline ever approved, which is neither.
"""

from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
    """How much of the question the answer covers, permissive to restrictive."""

    FULL = "full"
    """The answer covers everything that was asked."""

    PARTIAL = "partial"
    """The answer covers some of it and leaves part of the question unanswered."""

    NONE = "none"
    """The answer does not address what was asked at all."""

    def __str__(self) -> str:  # so f-strings and CLI output read as "none", not "Verdict.NONE"
        return self.value


#: Permissive first. Index in this tuple is the severity of a verdict.
ORDER: tuple[Verdict, ...] = (Verdict.FULL, Verdict.PARTIAL, Verdict.NONE)

#: The verdict every default resolves to. See the module docstring.
PERMISSIVE: Verdict = Verdict.FULL


def severity(verdict: Verdict) -> int:
    """Position on the scale. Higher means the judge is less happy with the answer."""
    return ORDER.index(Verdict(verdict))


def is_degradation(before: Verdict, after: Verdict) -> bool:
    """True when moving `before` -> `after` withholds more than it did."""
    return severity(after) > severity(before)


def clamp_to_degradation(original: Verdict, proposed: Verdict) -> Verdict:
    """Return `proposed` only if it degrades; otherwise keep `original`.

    This is the enforcement point for the module's invariant. Route every
    verdict a judge produces through here and a judge becomes structurally
    incapable of promoting a turn, no matter what the model returned, what the
    prompt said, or how the caller wired it up.
    """
    original, proposed = Verdict(original), Verdict(proposed)
    return proposed if is_degradation(original, proposed) else original


def parse(raw: object) -> Verdict | None:
    """Parse a verdict from model output. Unrecognised input is `None`, never a guess.

    An unreadable field must resolve to the safe outcome at the call site rather
    than be coerced here into a decision nobody made.
    """
    if isinstance(raw, Verdict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return Verdict(raw.strip().lower())
    except ValueError:
        return None
