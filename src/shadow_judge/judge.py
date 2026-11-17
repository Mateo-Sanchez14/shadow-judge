"""The judge: one bounded question, asked by a model that is not the one being judged.

Three design rules are baked in here, and each one is a decision you would
otherwise have to make under pressure later:

**The question is narrow.** A judge that re-runs your pipeline's own
instructions buys a second opinion with correlated errors, which is no opinion
at all. The prompt here asks one thing — does this answer cover what was asked
— and deliberately refuses to judge tone, length or style. Keep it that way: the
value is decomposition, not a second pass.

**The model is independent.** If the judge resolves its model from the same
setting as the pipeline it judges, the two move together and you can never
measure a model change against anything. `Judge` takes its own provider, and
`ShadowJudgeError` is raised if you hand it the same provider instance that
produced the answers.

**Failure is a verdict, not an exception.** `Judge.evaluate` never raises. A
provider outage resolves to `on_failure` (by default the permissive verdict), so
a check that could not run is simply no check and the turn behaves exactly as it
did before the check existed. The `failed` flag is what keeps that from being
invisible: a pipeline quietly degrading itself because its safety net is down
must never look like a pipeline working as designed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .verdict import PERMISSIVE, Verdict, parse

#: Never more than this many characters of any single answer reach the judge.
#: Far above a typical preview budget on purpose — this judges what the user
#: actually receives — but still bounded, because an operator can paste an
#: arbitrarily long answer into a knowledge base.
MAX_ANSWER_CHARS = 8000

#: The whole output is one short JSON object; the cap only stops a misbehaving
#: completion from running long.
MAX_OUTPUT_TOKENS = 30

DEFAULT_PROMPT = (
    "You check one thing about an answer an assistant is about to send, and "
    "nothing else. You are given a user's message and the answer, which was "
    "written in advance by a human and selected by an automated system. Decide "
    "how much of what the user asked for that answer actually contains.\n"
    'Reply with a JSON object of the form {"verdict": "full"|"partial"|"none"}:\n'
    '- "full": the answer covers everything the user asked. A refusal, or a '
    "referral elsewhere, settles a question as completely as a yes does — but it "
    "has to refuse THE question that was asked. An answer turning down something "
    'the user never brought up is "none".\n'
    '- "partial": the answer covers some of it and leaves part of the message '
    "unanswered — they asked about three things and it covers one, or they asked "
    "the price and the availability and it gives only the price.\n"
    '- "none": the answer does not address what was asked at all. This includes '
    "an answer about a NEIGHBOURING subject or category, however similar it "
    "sounds, and a list that does not contain the specific item the user named.\n"
    "Judge only whether the content matches the question. Do NOT judge tone, "
    "length, formatting, or whether you would have written it differently. A "
    'terse but correct answer is "full". When the answer plainly addresses the '
    "question, say so — this check exists to catch answers about the wrong "
    "thing, not to second-guess good ones."
)


class ShadowJudgeError(Exception):
    """Raised for wiring mistakes, never for a judging failure."""


class Provider(Protocol):
    """The one call a judge needs from a model provider.

    Returns the raw text of the completion. Implementations live in
    `shadow_judge.providers`; anything matching this shape will do.
    """

    def complete(self, system: str, user: str, *, max_output_tokens: int) -> str: ...


@dataclass
class Judgement:
    """What the judge decided, and whether it actually got to decide."""

    verdict: Verdict = PERMISSIVE
    failed: bool = False
    """True when `verdict` is a fallback rather than a judgment."""
    reason: str | None = None
    raw: str | None = None
    usage: dict = field(default_factory=dict)

    @property
    def judged(self) -> bool:
        """True when a real verdict came back from the model."""
        return not self.failed


def build_prompt(message: str, answers: list[str]) -> str:
    """The per-turn context: the question, then the answer as the user sees it."""
    lines = [f'User\'s message:\n"{message.strip()}"', "", "The answer about to be sent:"]
    # A turn can send several answers, each as its own message; the user
    # experiences them as one reply, so the judge weighs them together.
    for index, answer in enumerate(answers, 1):
        prefix = f"[{index}] " if len(answers) > 1 else ""
        lines.append(f"{prefix}{_clip(answer)}")
    return "\n".join(lines)


def _clip(answer: str) -> str:
    text = answer.strip()
    return text if len(text) <= MAX_ANSWER_CHARS else text[:MAX_ANSWER_CHARS] + "…"


def parse_response(content: str) -> tuple[Verdict | None, str | None]:
    """`(verdict, error)`. Anything unrecognised is an error, never a guess."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None, f"judge returned unparseable content: {content[:120]!r}"
    raw = data.get("verdict") if isinstance(data, dict) else None
    verdict = parse(raw)
    if verdict is None:
        return None, f"judge returned an unknown verdict: {raw!r}"
    return verdict, None


class Judge:
    """Asks one bounded question about an answer, and never raises doing it."""

    def __init__(
        self,
        provider: Provider,
        *,
        prompt: str = DEFAULT_PROMPT,
        on_failure: Verdict = PERMISSIVE,
        judged_provider: Provider | None = None,
    ):
        """
        Args:
            provider: the model that judges. Give it its own client with a
                short timeout and no retries: this sits on the critical path,
                and the whole point of the failure path is that it gives up fast.
            prompt: override only to narrow the question further, never to widen
                it into a second pass over your pipeline's own instructions.
            on_failure: the verdict a failed call resolves to. The default is
                the permissive one so an outage is a no-op rather than a bot
                that goes silent for the duration.
            judged_provider: the provider that produced the answers, if you have
                it here. Passing the same object as `provider` is a wiring
                mistake and raises: a judge sharing a model with what it judges
                moves with it and can never measure it.
        """
        if judged_provider is not None and judged_provider is provider:
            raise ShadowJudgeError(
                "The judge must not share a provider with the pipeline it judges: "
                "they would move together and a model change could never be "
                "measured against anything. Build a second provider instance."
            )
        self.provider = provider
        self.prompt = prompt
        self.on_failure = Verdict(on_failure)

    def evaluate(self, message: str, answers: list[str]) -> Judgement:
        """Judge the answer about to go out. Never raises — failure is a verdict."""
        if not answers:
            # Nothing was going to be sent, so there is nothing to withhold.
            # Not a call, not a failure — just no work.
            return Judgement()

        try:
            content = self.provider.complete(
                self.prompt,
                build_prompt(message, answers),
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:  # the safety net must not become a new way to fail
            return Judgement(
                verdict=self.on_failure,
                failed=True,
                reason=f"judge unavailable ({type(exc).__name__}: {exc})",
            )

        verdict, error = parse_response(content or "")
        if verdict is None:
            return Judgement(
                verdict=self.on_failure, failed=True, reason=error, raw=content
            )
        return Judgement(
            verdict=verdict,
            raw=content,
            usage=dict(getattr(self.provider, "last_usage", {}) or {}),
        )
