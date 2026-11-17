import pytest

from shadow_judge import FakeProvider, Judge, Turn, Verdict, replay


def turn(i, *, answered=True, escalated=False):
    return Turn(
        id=f"t{i}",
        message=f"message {i}",
        answers=[f"answer {i}"] if answered else [],
        escalated=escalated,
    )


def test_replay_judges_every_turn_once_per_run():
    turns = [turn(i) for i in range(4)]
    provider = FakeProvider("full")
    report = replay(turns, Judge(provider), runs=3)

    assert len(provider.calls) == 12
    assert report.turns == 4
    assert report.runs == 3
    assert all(len(r.judgements) == 3 for r in report.results)


def test_replay_changes_nothing():
    """The corpus is input. A replay must not touch it."""
    turns = [turn(0, escalated=True), turn(1)]
    before = [(t.message, tuple(t.answers), t.escalated) for t in turns]
    replay(turns, Judge(FakeProvider("none")), runs=2)
    after = [(t.message, tuple(t.answers), t.escalated) for t in turns]
    assert before == after


def test_unanswered_turns_cost_no_judge_call():
    provider = FakeProvider("none")
    report = replay([turn(0, answered=False), turn(1)], Judge(provider), runs=1)
    assert len(provider.calls) == 1
    assert report.turns == 2
    assert report.answered == 1


def test_runs_must_be_positive():
    with pytest.raises(ValueError):
        replay([turn(0)], Judge(FakeProvider()), runs=0)


def test_instability_is_reported_as_a_number():
    """One run proves nothing; disagreement has to show up as a figure."""
    # Same turn judged full, then none: it disagreed with itself.
    provider = FakeProvider(["full", "none"])
    report = replay([turn(0)], Judge(provider), runs=2)

    assert len(report.unstable) == 1
    assert report.stability == 0.0
    assert report.unstable[0].verdicts == [Verdict.FULL, Verdict.NONE]


def test_a_single_run_reports_no_stability_rather_than_a_fake_one():
    report = replay([turn(0)], Judge(FakeProvider("full")), runs=1)
    assert report.stability is None
    assert report.unstable == []


def test_majority_settles_a_disagreement():
    provider = FakeProvider(["none", "full", "full"])
    report = replay([turn(0)], Judge(provider), runs=3)
    assert report.results[0].settled("majority") is Verdict.FULL


def test_a_tie_settles_toward_the_permissive_end():
    """A judge that could not make up its mind has not earned a withholding."""
    provider = FakeProvider(["none", "full"])
    report = replay([turn(0)], Judge(provider), runs=2)
    assert report.results[0].settled("majority") is Verdict.FULL


def test_worst_takes_the_pessimistic_bound():
    provider = FakeProvider(["full", "full", "none"])
    report = replay([turn(0)], Judge(provider), runs=3)
    assert report.results[0].settled("worst") is Verdict.NONE
    assert report.results[0].settled("first") is Verdict.FULL


def test_failures_are_counted_across_runs():
    class Down:
        model = "down"

        def complete(self, system, user, *, max_output_tokens):
            raise ConnectionError("nope")

    report = replay([turn(0), turn(1)], Judge(Down()), runs=2)
    assert report.failures == 4
