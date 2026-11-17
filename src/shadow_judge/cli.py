"""`shadow-judge replay` — run a corpus through a judge and print the arithmetic.

    shadow-judge replay --corpus turns.json --model gpt-4.1-mini --runs 3
    shadow-judge replay --corpus turns.json --provider anthropic --model claude-haiku-4-5
    shadow-judge replay --corpus examples/corpus.json --provider fake --runs 3

The `fake` provider needs no API key and no network, so the examples run and the
output shape can be inspected before anyone spends a token.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import CorpusError, load
from .judge import Judge
from .providers import AnthropicProvider, FakeProvider, OpenAIProvider
from .report import ShadowReport
from .shadow import replay
from .verdict import ORDER, Verdict


#: Substrings of the shipped example corpus, and what a competent judge should
#: say about each. Keyed on the planted cases so `--provider fake` demonstrates
#: the report rather than filling it with noise. This is a scripted stand-in,
#: not a model: it knows the answers because they were written down next to the
#: questions in `examples/corpus.json`.
_DEMO_CASES = (
    ("magsafe cases", "none"),      # answered about a neighbouring category
    ("12 instalments", "none"),     # asked instalments, told payment methods
    ("do you have it in blue", "partial"),  # two questions, one answered
    ("trade-ins", "full"),          # a refusal settles the question
)

#: Turns a judge can legitimately read either way. Alternating them is how the
#: demo shows that one run proves nothing.
_DEMO_AMBIGUOUS = ("pencil in stock",)


def _demo_judge():
    """A reproducible stand-in judge for `--provider fake`, with no API key."""
    seen: dict[str, int] = {}

    def decide(system: str, user: str) -> str:  # noqa: ARG001
        text = user.lower()
        for needle, verdict in _DEMO_CASES:
            if needle in text:
                return verdict
        for needle in _DEMO_AMBIGUOUS:
            if needle in text:
                # "we carry X and Y" never confirmed stock. A real judge wavers
                # here too; this one wavers on a schedule so the run is stable.
                count = seen[needle] = seen.get(needle, 0) + 1
                return "partial" if count % 2 else "full"
        return "full"

    return decide


def build_provider(name: str, model: str | None, timeout: float, script: str | None):
    if name == "openai":
        return OpenAIProvider(model or "gpt-4.1-mini", timeout=timeout)
    if name == "anthropic":
        return AnthropicProvider(model or "claude-haiku-4-5", timeout=timeout)
    if name == "fake":
        if script:
            provider = FakeProvider([v.strip() for v in script.split(",")])
        else:
            # A seeded draw rather than a cycle: a cycle lines up with the corpus
            # and produces either perfect stability or none at all, neither of
            # which looks like a judge. This gives a plausible spread and the
            # kind of run-to-run disagreement the report exists to surface --
            # and the fixed seed keeps it reproducible.
            provider = FakeProvider(_demo_judge())
        provider.model = model or "fake"  # type: ignore[attr-defined]
        return provider
    raise SystemExit(f"unknown provider {name!r}; expected openai, anthropic or fake")


def as_json(report: ShadowReport) -> str:
    return json.dumps(
        {
            "turns": report.turns,
            "answered": report.answered,
            "runs": report.runs,
            "model": report.model,
            "settle": report.settle,
            "failures": report.failures,
            "stability": report.stability,
            "distribution": {str(k): v for k, v in report.distribution.items()},
            "unstable": [
                {"id": r.turn.id, "message": r.turn.message, "verdicts": [str(v) for v in r.verdicts]}
                for r in report.unstable
            ],
            "impacts": [
                {
                    "acted_on": [str(v) for v in i.acted_on],
                    "answered": i.answered,
                    "escalated_before": i.escalated_before,
                    "escalated_after": i.escalated_after,
                    "withheld": i.withheld,
                    "multiplier": i.multiplier,
                }
                for i in report.impacts()
            ],
            "usage": report.usage,
        },
        indent=2,
        ensure_ascii=False,
    )


def print_detail(report: ShadowReport) -> None:
    """Every turn the judge was not happy with, worst first."""
    for verdict in reversed(ORDER):
        if verdict is Verdict.FULL:
            continue
        results = report.of(verdict)
        if not results:
            continue
        print(f"\n--- {str(verdict).upper()} ({len(results)}) ---")
        for result in results:
            marker = "" if result.unanimous else "  [unstable: %s]" % ", ".join(
                str(v) for v in result.verdicts
            )
            label = result.turn.id or "?"
            print(f"\n[{label}] {result.turn.message[:100]!r}{marker}")
            for answer in result.turn.answers:
                print(f"    -> {answer[:160]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shadow-judge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("replay", help="replay a corpus through a judge")
    run.add_argument("--corpus", required=True, help="path to the corpus JSON")
    run.add_argument(
        "--provider", default="openai", choices=["openai", "anthropic", "fake"]
    )
    run.add_argument("--model", default=None, help="model for the judge")
    run.add_argument(
        "--runs",
        type=int,
        default=1,
        help="replay the whole corpus N times; >1 measures how unstable the judge is",
    )
    run.add_argument(
        "--settle",
        default="majority",
        choices=["majority", "first", "worst"],
        help="how to pick one verdict per turn when runs disagree",
    )
    run.add_argument("--timeout", type=float, default=10.0)
    run.add_argument("--json", action="store_true", help="machine-readable output")
    run.add_argument("--detail", action="store_true", help="print every non-full turn")
    run.add_argument("--quiet", action="store_true", help="no progress output")
    run.add_argument(
        "--fake-verdicts",
        default=None,
        help="comma-separated verdicts to cycle through with --provider fake",
    )

    args = parser.parse_args(argv)

    try:
        turns = load(args.corpus)
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    provider = build_provider(args.provider, args.model, args.timeout, args.fake_verdicts)
    judge = Judge(provider)

    progress = None
    if not args.quiet and not args.json:
        total = len(turns) * args.runs
        state = {"done": 0}

        def progress(run_index, turn_index, turn, judgement):  # noqa: ARG001
            state["done"] += 1
            print(f"\r  judging {state['done']}/{total}...", end="", file=sys.stderr)

    report = replay(turns, judge, runs=args.runs, settle=args.settle, on_progress=progress)
    if progress is not None:
        print("\r" + " " * 32 + "\r", end="", file=sys.stderr)

    if args.json:
        print(as_json(report))
    else:
        print(report.summary())
        if args.detail:
            print_detail(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
