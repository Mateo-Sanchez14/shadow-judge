import pytest

from shadow_judge import (
    DEFAULT_PROMPT,
    FakeProvider,
    Judge,
    ShadowJudgeError,
    Verdict,
    build_prompt,
)
from shadow_judge.judge import MAX_ANSWER_CHARS, parse_response


class Boom:
    """A provider that is down."""

    model = "boom"

    def complete(self, system, user, *, max_output_tokens):
        raise TimeoutError("upstream is not answering")


class Garbage:
    model = "garbage"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user, *, max_output_tokens):
        return self.payload


def test_a_working_judge_returns_the_models_verdict():
    judge = Judge(FakeProvider("none"))
    result = judge.evaluate("do you sell cases?", ["Our charging cubes are 20 USD."])
    assert result.verdict is Verdict.NONE
    assert result.judged and not result.failed


def test_a_provider_outage_is_a_verdict_not_an_exception():
    judge = Judge(Boom())
    result = judge.evaluate("anything", ["an answer"])
    assert result.failed
    assert result.verdict is Verdict.FULL, "an outage must be a no-op, not a silent gag"
    assert "TimeoutError" in result.reason


def test_the_failure_verdict_is_configurable():
    judge = Judge(Boom(), on_failure=Verdict.PARTIAL)
    assert judge.evaluate("anything", ["an answer"]).verdict is Verdict.PARTIAL


@pytest.mark.parametrize(
    "payload",
    ['{"verdict": "maybe"}', "not json at all", "", '{"other": 1}', "[]"],
)
def test_unreadable_output_fails_loudly_rather_than_guessing(payload):
    result = Judge(Garbage(payload)).evaluate("anything", ["an answer"])
    assert result.failed
    assert result.reason
    assert result.verdict is Verdict.FULL


def test_no_answers_means_no_call_and_no_failure():
    provider = FakeProvider("none")
    result = Judge(provider).evaluate("a message nobody answered", [])
    assert result.verdict is Verdict.FULL
    assert not result.failed
    assert provider.calls == [], "nothing was going to be sent, so nothing was judged"


def test_sharing_a_provider_with_the_judged_pipeline_is_refused():
    shared = FakeProvider("full")
    with pytest.raises(ShadowJudgeError, match="must not share a provider"):
        Judge(shared, judged_provider=shared)


def test_a_separate_provider_instance_is_fine():
    Judge(FakeProvider("full"), judged_provider=FakeProvider("full"))


def test_the_default_prompt_refuses_to_judge_style():
    """The judge's value is a narrow question. Widening it is the failure mode."""
    lowered = DEFAULT_PROMPT.lower()
    assert "do not judge tone" in lowered
    assert "formatting" in lowered
    for verdict in ("full", "partial", "none"):
        assert f'"{verdict}"' in DEFAULT_PROMPT


def test_prompt_carries_the_message_and_every_answer():
    prompt = build_prompt("  can you ship to Córdoba?  ", ["three days", "by courier"])
    assert "can you ship to Córdoba?" in prompt
    assert "[1] three days" in prompt
    assert "[2] by courier" in prompt


def test_a_single_answer_is_not_numbered():
    assert "[1]" not in build_prompt("hi", ["just the one"])


def test_long_answers_are_clipped():
    prompt = build_prompt("hi", ["x" * (MAX_ANSWER_CHARS + 500)])
    assert len(prompt) < MAX_ANSWER_CHARS + 200
    assert prompt.endswith("…")


def test_parse_response_reports_the_offending_payload():
    verdict, error = parse_response('{"verdict": "sometimes"}')
    assert verdict is None
    assert "sometimes" in error
