"""Replay: run a corpus of real turns through the judge without letting it act.

This is the whole point of the library. Before a judge is allowed to withhold
anything, you replay what your pipeline already did and count what it *would*
have withheld. If that number is zero, the judge is latency and failure surface
for nothing and should not ship. If it is large, you have found out here rather
than from the person who had to work through the backlog it created.

Nothing in this module sends, blocks, or mutates anything. It reads a corpus and
returns arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .corpus import Turn
from .judge import Judge, Judgement
from .providers import usage_total
from .report import ShadowReport, Settle, TurnResult

Progress = Callable[[int, int, Turn, Judgement], None]


def replay(
    turns: Sequence[Turn],
    judge: Judge,
    *,
    runs: int = 1,
    settle: Settle = "majority",
    on_progress: Progress | None = None,
) -> ShadowReport:
    """Judge every turn `runs` times and return what it would have changed.

    Args:
        turns: turns your pipeline already handled. See `shadow_judge.corpus`.
        judge: the judge under test. It is never given the chance to act.
        runs: how many times to replay the whole corpus. More than one is the
            point: these models are not deterministic even at temperature 0, and
            a single pass gives you a number with no error bar. Three is a
            sensible floor for a decision you intend to defend.
        settle: how to pick one verdict per turn once the runs disagree.
        on_progress: called as `(run_index, turn_index, turn, judgement)` after
            every call, for a progress bar or a live log.

    Returns:
        A `ShadowReport`. Nothing was sent, blocked or changed.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")

    collected: list[list[Judgement]] = [[] for _ in turns]
    usages: list[dict] = []

    for run_index in range(runs):
        for turn_index, turn in enumerate(turns):
            judgement = judge.evaluate(turn.message, list(turn.answers))
            collected[turn_index].append(judgement)
            if judgement.usage:
                usages.append(judgement.usage)
            if on_progress is not None:
                on_progress(run_index, turn_index, turn, judgement)

    return ShadowReport(
        results=[
            TurnResult(turn=turn, judgements=judgements)
            for turn, judgements in zip(turns, collected)
        ],
        runs=runs,
        settle=settle,
        model=getattr(judge.provider, "model", None),
        usage=usage_total(usages),
    )
