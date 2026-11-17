"""The replay corpus: turns your pipeline has already handled.

This is the part people skip, and it is the part that makes the rest worth
anything. A judge evaluated against cases you wrote by hand tells you the judge
agrees with you. A judge replayed over what your pipeline actually sent tells
you what it would have changed — which is the only number that decides whether
the thing ships.

So the corpus is an export, not a fixture. Every turn is something that really
happened:

```json
{
  "version": 1,
  "turns": [
    {
      "id": "msg-123",
      "message": "do you ship to Córdoba?",
      "answers": ["We ship nationwide, 3-5 business days."],
      "escalated": false
    }
  ]
}
```

`escalated` is the field that carries the cost. It records whether that turn
already reached a human. Without it you can still count verdicts; with it you
can answer the question a judge actually has to justify — what does acting on
this verdict do to the pile of work on a person's desk.

A bare JSON list of turns is accepted too, so an export that predates the
envelope still loads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CORPUS_VERSION = 1


@dataclass(frozen=True)
class Turn:
    """One thing that already happened."""

    message: str
    """What the user said."""

    answers: list[str] = field(default_factory=list)
    """What the pipeline sent back. Empty means it sent nothing."""

    id: str | None = None
    escalated: bool = False
    """True when this turn already reached a human, before any judge existed."""

    context: dict[str, Any] = field(default_factory=dict)
    """Anything else from the export. Carried through to the report, never judged."""

    @property
    def answered(self) -> bool:
        """True when the pipeline actually sent something a judge could withhold."""
        return bool(self.answers)


class CorpusError(ValueError):
    """The corpus file is not something we can replay."""


def from_dicts(rows: list[dict]) -> list[Turn]:
    """Build turns from raw dictionaries, keeping unknown keys in `context`."""
    known = {"message", "answers", "id", "escalated"}
    turns: list[Turn] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CorpusError(f"turn {index} is {type(row).__name__}, expected an object")
        message = row.get("message")
        if not isinstance(message, str) or not message.strip():
            raise CorpusError(f"turn {index} has no usable 'message'")
        raw_answers = row.get("answers", [])
        if isinstance(raw_answers, str):  # a single answer is a common export shape
            raw_answers = [raw_answers]
        if not isinstance(raw_answers, list) or any(
            not isinstance(a, str) for a in raw_answers
        ):
            raise CorpusError(f"turn {index} has an 'answers' that is not a list of strings")
        turns.append(
            Turn(
                message=message,
                answers=list(raw_answers),
                id=str(row["id"]) if row.get("id") is not None else None,
                escalated=bool(row.get("escalated", False)),
                context={k: v for k, v in row.items() if k not in known},
            )
        )
    return turns


def load(path: str | Path) -> list[Turn]:
    """Read a corpus file. Accepts the envelope or a bare list of turns."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CorpusError(f"no corpus at {path}") from exc
    except ValueError as exc:
        raise CorpusError(f"{path} is not valid JSON: {exc}") from exc

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get("turns"), list):
        rows = data["turns"]
    else:
        raise CorpusError(
            f"{path} must hold a list of turns, or an object with a 'turns' list"
        )
    if not rows:
        raise CorpusError(f"{path} holds no turns")
    return from_dicts(rows)


def dump(turns: list[Turn], path: str | Path) -> None:
    """Write a corpus file in the current envelope."""
    rows = [
        {
            **({"id": t.id} if t.id is not None else {}),
            "message": t.message,
            "answers": list(t.answers),
            "escalated": t.escalated,
            **t.context,
        }
        for t in turns
    ]
    body = {"version": CORPUS_VERSION, "turns": rows}
    Path(path).write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
