"""shadow-judge: measure what an LLM judge would have changed, before it changes it.

```python
from shadow_judge import Judge, OpenAIProvider, Verdict, load, replay

turns = load("turns.json")                       # what your pipeline already did
judge = Judge(OpenAIProvider("gpt-4.1-mini"))    # its own model, its own budget
report = replay(turns, judge, runs=3)            # nothing is sent, blocked or changed

print(report.summary())
print(report.impact(Verdict.NONE))               # the number you decide from
```
"""

from .corpus import CORPUS_VERSION, CorpusError, Turn, dump, from_dicts, load
from .judge import (
    DEFAULT_PROMPT,
    Judge,
    Judgement,
    Provider,
    ShadowJudgeError,
    build_prompt,
)
from .providers import AnthropicProvider, FakeProvider, OpenAIProvider
from .report import ShadowReport, TurnResult, WorkloadImpact
from .shadow import replay
from .verdict import ORDER, PERMISSIVE, Verdict, clamp_to_degradation, is_degradation

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider",
    "CORPUS_VERSION",
    "CorpusError",
    "DEFAULT_PROMPT",
    "FakeProvider",
    "Judge",
    "Judgement",
    "ORDER",
    "OpenAIProvider",
    "PERMISSIVE",
    "Provider",
    "ShadowJudgeError",
    "ShadowReport",
    "Turn",
    "TurnResult",
    "Verdict",
    "WorkloadImpact",
    "build_prompt",
    "clamp_to_degradation",
    "dump",
    "from_dicts",
    "is_degradation",
    "load",
    "replay",
]
