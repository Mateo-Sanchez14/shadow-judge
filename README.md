# shadow-judge

**Measure what an LLM judge would have changed, before you let it change anything.**

Adding a judge in front of your pipeline is easy. Deciding whether to let it
*act* is the hard part, and it is not a question a prompt can answer. This
library replays turns your pipeline already handled, asks the judge what it
would have done, and reports the answer in the only unit the decision is
actually made in: **how much work lands on a person's desk.**

Run it on the shipped example — no API key, no network:

```console
$ shadow-judge replay --corpus examples/corpus.json --provider fake --runs 3
10 turns, 8 answered, 3 run(s), model=fake

verdicts (settled by majority):
  full         4   50.0%
  partial      2   25.0%
  none         2   25.0%

stability: 90.0% of turns agreed with themselves across 3 runs (1 disagreed)

if the judge acts:
  acting on none: flagged turns 1 -> 2 per 8 answered (2.0x)
  acting on none, partial: flagged turns 1 -> 4 per 8 answered (4.0x)
```

Those last two lines are the whole library. At real volumes, one of them is a
feature and the other is a person quitting.

## Why this exists

This is extracted from a WhatsApp and Instagram assistant running in production,
where a judge was built to catch a specific failure: the classifier picks the
right FAQ and its stored answer still does not cover what the customer asked
("MagSafe cases" answered with the charging-cube FAQ).

Before wiring it into live traffic, it was replayed over a month of real
replies. The numbers said: act on `none`, and leave `partial` recorded but
inert — because acting on `partial` too would have taken flagged threads **from
4 to 29 per 56 answered turns**. The judge was worth shipping. Half of it was
not, and that half would have been shipped on vibes.

The replay cost a few cents. Finding out the other way costs a salesperson their
week.

## Install

Not on PyPI yet — install from the repository:

```bash
pip install 'git+https://github.com/Mateo-Sanchez14/shadow-judge'
pip install 'shadow-judge[openai] @ git+https://github.com/Mateo-Sanchez14/shadow-judge'
```

The core has no dependencies; `openai` and `anthropic` are extras you only need
for the adapter you actually use.

## Use it

```python
from shadow_judge import Judge, OpenAIProvider, Verdict, load, replay

turns  = load("turns.json")                    # what your pipeline already did
judge  = Judge(OpenAIProvider("gpt-4.1-mini")) # its own model, its own budget
report = replay(turns, judge, runs=3)          # nothing is sent, blocked or changed

print(report.summary())
print(report.impact(Verdict.NONE))             # the number you decide from
```

Or from the shell, with no API key and no network:

```bash
shadow-judge replay --corpus examples/corpus.json --provider fake --runs 3
```

### The corpus is an export, not a fixture

This is the part people skip, and it is the part that makes the rest worth
anything. A judge evaluated against cases you wrote by hand tells you the judge
agrees with you.

```json
{
  "version": 1,
  "turns": [
    {
      "id": "msg-123",
      "message": "do you have MagSafe cases for the 15 Pro?",
      "answers": ["Our charging cubes come in 20W and 30W, both in stock."],
      "escalated": false
    }
  ]
}
```

`escalated` is the field that carries the cost: it records whether that turn
already reached a human. Without it you can count verdicts. With it you can
answer the question a judge actually has to justify.

## The three rules it enforces

**A judge can only degrade.** There is no verdict that turns a withheld answer
into a sent one. `clamp_to_degradation` is the enforcement point, and
`test_clamp_never_promotes_for_any_pair` checks it exhaustively. That asymmetry
is what makes a judge safe to switch on: the worst a broken one can do is hold
back answers that were fine, which is visible and recoverable. A judge that
could also promote would be able to send an answer your pipeline never approved,
which is neither.

**The judge's model is not the judged model.** Sharing one means the two move
together and a model change can never be measured against anything. Pass
`judged_provider=` and the library refuses the wiring outright.

**Failure is a verdict, not an exception.** `Judge.evaluate` never raises. An
outage resolves to the permissive verdict, so a check that could not run is
simply no check and the turn behaves exactly as it did before the check existed.
The `failed` flag is what keeps that from being invisible — a pipeline quietly
degrading itself because its safety net is down must never look like a pipeline
working as designed.

## One run proves nothing

These models are not deterministic even at temperature 0. A single pass gives
you a number with no error bar, which is worse than no number because it looks
like one.

`runs=3` replays the whole corpus three times, settles each turn by majority
(ties break toward the permissive end — a judge that could not make up its mind
has not earned a withholding) and reports every turn that disagreed with itself.
Instability becomes a figure in the report instead of a surprise in production.

Use `settle="worst"` for the pessimistic bound: what an unlucky day looks like.

## What it does not do

- **It does not decide for you.** It prints the arithmetic. Whether 2.5x the
  flagged threads is worth catching six wrong answers is a question about your
  team, not your model.
- **It does not judge quality, tone or style.** The built-in prompt asks one
  bounded question — does this answer cover what was asked — and explicitly
  refuses the rest. Widening it back out is the failure mode it was factored to
  avoid: a judge that re-runs your pipeline's own instructions buys a second
  opinion with correlated errors, which is no opinion at all.
- **It does not need your pipeline.** It only ever sees `(message, answers)`, so
  it works against any stack in any language as long as you can export the turns.
- **It is not a retrieval or classifier eval.** It judges the answer that was
  chosen, not the choosing. If your pipeline fails by *selection*, measure that
  first — this will not see it.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

## License

MIT
