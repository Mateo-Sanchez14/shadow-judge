"""The arithmetic the decision is made from.

The shape under test is the one that killed a feature in production: a judge
whose `none` verdict was worth acting on and whose `partial` verdict, on the
same corpus, would have multiplied the human workload several times over.
"""

from shadow_judge import FakeProvider, Judge, Turn, Verdict, replay
from shadow_judge.report import ShadowReport, TurnResult
from shadow_judge.judge import Judgement


def build(verdicts, *, escalated=(), answered=True):
    """One turn per verdict, judged once, with `escalated` marking turn indices."""
    results = []
    for index, verdict in enumerate(verdicts):
        turn = Turn(
            id=f"t{index}",
            message=f"m{index}",
            answers=[f"a{index}"] if answered else [],
            escalated=index in escalated,
        )
        results.append(TurnResult(turn=turn, judgements=[Judgement(verdict=verdict)]))
    return ShadowReport(results=results, runs=1)


def test_distribution_counts_only_answered_turns():
    report = build([Verdict.FULL, Verdict.NONE, Verdict.PARTIAL])
    assert report.distribution == {Verdict.FULL: 1, Verdict.PARTIAL: 1, Verdict.NONE: 1}


def test_the_decision_number():
    """56 answered turns, 4 already on a person's desk.

    Acting on `none` adds a handful. Acting on `partial` as well buries them.
    That difference is the entire reason to replay before shipping.
    """
    verdicts = (
        [Verdict.FULL] * 36 + [Verdict.NONE] * 6 + [Verdict.PARTIAL] * 14
    )
    assert len(verdicts) == 56
    # Four of the FULL turns had already gone to a human for other reasons.
    report = build(verdicts, escalated=(0, 1, 2, 3))

    on_none = report.impact(Verdict.NONE)
    assert on_none.answered == 56
    assert on_none.escalated_before == 4
    assert on_none.withheld == 6
    assert on_none.escalated_after == 10

    on_both = report.impact([Verdict.NONE, Verdict.PARTIAL])
    assert on_both.withheld == 20
    assert on_both.escalated_after == 24
    assert on_both.multiplier == 6.0

    assert "4 -> 10 per 56 answered" in str(on_none)


def test_already_escalated_turns_are_never_counted_twice():
    """A turn that already reached a human cannot be sent there again."""
    report = build([Verdict.NONE, Verdict.NONE], escalated=(0,))
    impact = report.impact(Verdict.NONE)
    assert impact.escalated_before == 1
    assert impact.withheld == 1, "only the turn that was NOT already escalated"
    assert impact.escalated_after == 2


def test_unanswered_turns_are_outside_the_denominator():
    report = build([Verdict.NONE, Verdict.NONE], answered=False)
    impact = report.impact(Verdict.NONE)
    assert impact.answered == 0
    assert impact.withheld == 0


def test_multiplier_is_none_when_nothing_was_escalated_before():
    report = build([Verdict.NONE])
    assert report.impact(Verdict.NONE).multiplier is None
    assert "x)" not in str(report.impact(Verdict.NONE))


def test_impacts_offers_the_two_choices_anyone_weighs():
    report = build([Verdict.NONE, Verdict.PARTIAL])
    cheap, expensive = report.impacts()
    assert cheap.acted_on == (Verdict.NONE,)
    assert expensive.acted_on == (Verdict.NONE, Verdict.PARTIAL)
    assert expensive.withheld >= cheap.withheld


def test_of_selects_turns_by_settled_verdict():
    report = build([Verdict.FULL, Verdict.NONE, Verdict.NONE])
    assert [r.turn.id for r in report.of(Verdict.NONE)] == ["t1", "t2"]


def test_summary_reads_as_a_decision_not_a_dump():
    report = build([Verdict.FULL] * 3 + [Verdict.NONE], escalated=(0,))
    text = report.summary()
    assert "4 turns, 4 answered" in text
    assert "if the judge acts:" in text
    assert "flagged turns 1 -> 2 per 4 answered" in text


def test_summary_survives_a_corpus_with_nothing_answered():
    report = build([Verdict.FULL], answered=False)
    assert "0 answered" in report.summary()


def test_end_to_end_through_replay():
    turns = [
        Turn(id="a", message="do you sell cases?", answers=["Cubes are 20 USD."]),
        Turn(id="b", message="shipping?", answers=["Three days."], escalated=True),
    ]
    report = replay(turns, Judge(FakeProvider(["none", "full"])), runs=1)
    assert report.distribution[Verdict.NONE] == 1
    assert report.impact(Verdict.NONE).escalated_after == 2
