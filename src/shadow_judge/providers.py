"""Model providers for the judge.

Each adapter builds its own client, because a judge needs a different budget
from the pipeline it judges: a short timeout and **no retries**. A retry doubles
the worst case for everything queued behind it, and the failure path already
resolves to a safe verdict quickly — retrying only delays that.

The SDKs are optional dependencies. Importing this module without them works;
only constructing the adapter you did not install will fail, and it will say so.

`FakeProvider` is part of the public API, not a test fixture. Pin a judge's
behaviour in your own tests with it, and run the examples with no API key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

#: A judge call is tiny. Anything slower than this is a stall, not a slow answer.
DEFAULT_TIMEOUT_SECONDS = 10.0


class OpenAIProvider:
    """OpenAI chat completions, pinned to JSON output."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        use_completion_tokens: bool | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "OpenAIProvider needs the openai package: pip install 'shadow-judge[openai]'"
            ) from exc

        kwargs = {"timeout": timeout, "max_retries": 0}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.last_usage: dict = {}
        # Newer model families reject `temperature` and renamed the output cap.
        # Left overridable because the split moves faster than this library.
        self._use_completion_tokens = (
            use_completion_tokens
            if use_completion_tokens is not None
            else _looks_like_reasoning_model(model)
        )

    def complete(self, system: str, user: str, *, max_output_tokens: int) -> str:
        cap = (
            {"max_completion_tokens": max_output_tokens}
            if self._use_completion_tokens
            else {"max_tokens": max_output_tokens, "temperature": 0}
        )
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **cap,
        )
        self.last_usage = _openai_usage(response)
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            return ""


class AnthropicProvider:
    """Anthropic messages, steered to JSON by prefilling the assistant turn."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "AnthropicProvider needs the anthropic package: "
                "pip install 'shadow-judge[anthropic]'"
            ) from exc

        kwargs = {"timeout": timeout, "max_retries": 0}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model
        self.last_usage: dict = {}

    def complete(self, system: str, user: str, *, max_output_tokens: int) -> str:
        response = self.client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_output_tokens,
            temperature=0,
            messages=[
                {"role": "user", "content": user},
                # Prefill: the reply continues this, so it cannot open with prose.
                {"role": "assistant", "content": "{"},
            ],
        )
        self.last_usage = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return "{" + text  # put back the prefilled brace


class FakeProvider:
    """Returns canned verdicts, in order. For tests, examples and dry runs.

    Pass a single verdict to answer everything the same way, a sequence to cycle
    through, or a callable taking `(system, user)` for anything smarter.

    A verdict word (`"none"`) is wrapped into the JSON a real provider would
    return; anything else a callable hands back is passed through untouched, so
    returning `"not json"` exercises the unparseable path and raising exercises
    the outage path.
    """

    def __init__(self, verdicts: object = "full"):
        self._callable = verdicts if callable(verdicts) else None
        if self._callable is None:
            if isinstance(verdicts, str):
                self._queue = [verdicts]
            elif isinstance(verdicts, Iterable):
                self._queue = [str(v) for v in verdicts] or ["full"]
            else:
                raise TypeError(f"cannot build a FakeProvider from {verdicts!r}")
        self._index = 0
        self.calls: list[tuple[str, str]] = []
        self.last_usage: dict = {"input_tokens": 0, "output_tokens": 0}

    def complete(self, system: str, user: str, *, max_output_tokens: int) -> str:
        self.calls.append((system, user))
        if self._callable is not None:
            return _as_completion(self._callable(system, user))
        verdict = self._queue[self._index % len(self._queue)]
        self._index += 1
        return f'{{"verdict": "{verdict}"}}'


def _as_completion(result: object) -> str:
    """Wrap a bare verdict word into provider-shaped JSON; pass anything else through."""
    from .verdict import parse

    if isinstance(result, str) and parse(result) is not None:
        return f'{{"verdict": "{result.strip().lower()}"}}'
    if isinstance(result, str):
        return result
    return f'{{"verdict": "{result}"}}'


def _looks_like_reasoning_model(model: str) -> bool:
    name = model.lower()
    return name.startswith(("o1", "o3", "o4")) or _major_version(name) >= 5


def _major_version(name: str) -> int:
    """Major version of a `gpt-<n>` style name, or 0 when there is not one."""
    if not name.startswith("gpt-"):
        return 0
    head = name[4:].split("-", 1)[0].split(".", 1)[0]
    return int(head) if head.isdigit() else 0


def _openai_usage(response: object) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "prompt_tokens_details", None)
    out = {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "cached_input_tokens": getattr(details, "cached_tokens", None),
    }
    return {k: v for k, v in out.items() if v is not None}


def usage_total(usages: Iterable[Mapping]) -> dict:
    """Sum a run's usage dictionaries, skipping keys nobody reported."""
    total: dict = {}
    for usage in usages:
        for key, value in (usage or {}).items():
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value
    return total
