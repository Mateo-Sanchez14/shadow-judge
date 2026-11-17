"""What the judge would have changed, in the units the decision is actually made in.

A verdict distribution is interesting. It is not a decision. The decision is
whether to let the judge act, and that is paid for by a person: every answer the
judge withholds is a thread that lands on someone's desk instead of resolving
itself. So the number that matters is not "31% partial", it is "acting on
partial takes the pile from 4 to 29 per 56 answered turns" — at which point the
answer is usually no, and you have learned that before shipping rather than from
the person whose desk it was.

Two things this module insists on:

**Settle across runs, do not trust one.** These models are not deterministic
even at temperature 0. A single pass over the corpus gives you a number with no
error bar. `ShadowReport` keeps every run and settles each turn by majority, and
`unstable` lists the turns that disagreed with themselves so instability shows
up as a number instead of as a surprise in production.

**Only count what could actually change.** A turn the pipeline never answered
has nothing to withhold, and a turn that already went to a human cannot be sent
there twice. Both are excluded from the impact, which is why the denominator is
`answered` and not `len(turns)`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Literal

from .corpus import Turn
from .judge import Judgement
from .verdict import ORDER, Verdict, severity

Settle = Literal["majority", "first", "worst"]


@dataclass(frozen=True)
class TurnResult:
    """Every verdict one turn produced, across every run."""

    turn: Turn
    judgements: list[Judgement]

    @property
    def verdicts(self) -> list[Verdict]:
        return [j.verdict for j in self.judgements]

    @property
    def unanimous(self) -> bool:
        return len(set(self.verdicts)) <= 1

    @property
    def failed_any(self) -> bool:
        return any(j.failed for j in self.judgements)

    def settled(self, how: Settle = "majority") -> Verdict:
        """The one verdict to attribute to this turn."""
        verdicts = self.verdicts
        if not verdicts:
            return Verdict.FULL
        if how == "first":
            return verdicts[0]
        if how == "worst":
            return max(verdicts, key=severity)
        counts = Counter(verdicts)
        top = max(counts.values())
        # Ties break toward the permissive end: a judge that could not make up
        # its mind has not earned a withholding.
        return next(v for v in ORDER if counts.get(v, 0) == top)


@dataclass(frozen=True)
class WorkloadImpact:
    """What acting on a set of verdicts does to the pile of human work."""

    acted_on: tuple[Verdict, ...]
    answered: int
    """Turns the pipeline answered — the only ones a judge could withhold."""
    escalated_before: int
    """Turns that already reached a human, before any judge existed."""
    withheld: int
    """Answered turns the judge would have held back, excluding ones already escalated."""

    @property
    def escalated_after(self) -> int:
        return self.escalated_before + self.withheld

    @property
    def multiplier(self) -> float | None:
        """How many times bigger the pile gets. `None` when it started empty."""
        if self.escalated_before == 0:
            return None
        return round(self.escalated_after / self.escalated_before, 1)

    def __str__(self) -> str:
        acted = ", ".join(str(v) for v in self.acted_on) or "nothing"
        head = (
            f"acting on {acted}: flagged turns {self.escalated_before} -> "
            f"{self.escalated_after} per {self.answered} answered"
        )
        return head if self.multiplier is None else f"{head} ({self.multiplier}x)"


@dataclass
class ShadowReport:
    """The outcome of a replay, and the arithmetic you decide from."""

    results: list[TurnResult]
    runs: int
    settle: Settle = "majority"
    model: str | None = None
    usage: dict = field(default_factory=dict)

    # -- what happened ----------------------------------------------------

    @property
    def turns(self) -> int:
        return len(self.results)

    @property
    def answered(self) -> int:
        return sum(1 for r in self.results if r.turn.answered)

    @property
    def failures(self) -> int:
        """Judge calls that could not produce a verdict, across every run."""
        return sum(1 for r in self.results for j in r.judgements if j.failed)

    @property
    def distribution(self) -> dict[Verdict, int]:
        """Settled verdict counts over the answered turns, permissive first."""
        counts = Counter(
            r.settled(self.settle) for r in self.results if r.turn.answered
        )
        return {v: counts.get(v, 0) for v in ORDER}

    @property
    def unstable(self) -> list[TurnResult]:
        """Turns whose verdict was not unanimous across runs."""
        if self.runs < 2:
            return []
        return [r for r in self.results if not r.unanimous]

    @property
    def stability(self) -> float | None:
        """Share of turns that agreed with themselves. `None` after a single run."""
        if self.runs < 2 or not self.results:
            return None
        return round(1 - len(self.unstable) / len(self.results), 3)

    def of(self, verdict: Verdict) -> list[TurnResult]:
        """Every turn that settled on `verdict`."""
        return [r for r in self.results if r.turn.answered and r.settled(self.settle) == verdict]

    # -- what it would cost -----------------------------------------------

    def impact(self, act_on: Iterable[Verdict] | Verdict) -> WorkloadImpact:
        """What happens to human workload if the judge acts on these verdicts."""
        acted = (
            (Verdict(act_on),)
            if isinstance(act_on, (str, Verdict))
            else tuple(Verdict(v) for v in act_on)
        )
        answered = [r for r in self.results if r.turn.answered]
        escalated_before = sum(1 for r in answered if r.turn.escalated)
        withheld = sum(
            1
            for r in answered
            if not r.turn.escalated and r.settled(self.settle) in acted
        )
        return WorkloadImpact(
            acted_on=acted,
            answered=len(answered),
            escalated_before=escalated_before,
            withheld=withheld,
        )

    def impacts(self) -> list[WorkloadImpact]:
        """The impact of each escalating option, cheapest first.

        `none` alone, then `none` and `partial` together — the two choices
        anyone actually weighs.
        """
        return [
            self.impact([Verdict.NONE]),
            self.impact([Verdict.NONE, Verdict.PARTIAL]),
        ]

    # -- how it reads ------------------------------------------------------

    def summary(self) -> str:
        lines = []
        head = f"{self.turns} turns, {self.answered} answered, {self.runs} run(s)"
        if self.model:
            head += f", model={self.model}"
        lines.append(head)

        total = self.answered or 1
        lines.append("")
        lines.append("verdicts (settled by %s):" % self.settle)
        for verdict, count in self.distribution.items():
            lines.append(f"  {str(verdict):<8} {count:>5}  {count / total:>6.1%}")

        if self.runs > 1:
            lines.append("")
            lines.append(
                f"stability: {self.stability:.1%} of turns agreed with themselves "
                f"across {self.runs} runs ({len(self.unstable)} disagreed)"
            )
        if self.failures:
            lines.append(
                f"failures:  {self.failures} judge call(s) could not produce a verdict"
            )

        lines.append("")
        lines.append("if the judge acts:")
        for impact in self.impacts():
            lines.append(f"  {impact}")

        if self.usage:
            lines.append("")
            lines.append(
                "tokens: " + ", ".join(f"{k}={v}" for k, v in sorted(self.usage.items()))
            )
        return "\n".join(lines)
