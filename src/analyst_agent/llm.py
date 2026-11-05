"""Thin wrapper over LiteLLM with tool calling, plus a scripted FakeLLM for tests.

Both classes have the same method, ``complete(messages, tools) -> LLMResponse``,
so the agent does not care which one it receives.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "mistral/mistral-small-latest"


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    name: str
    arguments: dict
    id: str = "call_0"


@dataclass
class LLMResponse:
    """What the model said in one turn, plus usage numbers."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None  # None when LiteLLM cannot price the model

    def assistant_message(self) -> dict:
        """This turn as an OpenAI-style message, to append to the conversation history."""
        message: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return message


class LiteLLMClient:
    """Calls any model supported by LiteLLM. The model name comes from the MODEL env variable."""

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self.model = model or os.getenv("MODEL", DEFAULT_MODEL)
        self.temperature = temperature

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        import litellm  # imported here so tests that use FakeLLM do not need it

        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            num_retries=2,
        )
        message = response.choices[0].message

        tool_calls = []
        for raw in message.tool_calls or []:
            try:
                arguments = json.loads(raw.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}  # the tool will answer "missing argument" and the model retries
            tool_calls.append(ToolCall(name=raw.function.name, arguments=arguments, id=raw.id))

        usage = getattr(response, "usage", None)
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:  # unknown model price: report no cost rather than fail
            cost = None

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
        )


class FakeLLM:
    """Returns pre-written responses in order. Used by the tests and offline demos."""

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls: list[list[dict]] = []  # the messages received at each turn

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            raise RuntimeError("FakeLLM script is exhausted.")
        return self.script.pop(0)


def tool_turn(*calls: tuple[str, dict]) -> LLMResponse:
    """Shortcut to script a turn where the model calls tools: tool_turn(("run_sql", {...}))."""
    return LLMResponse(
        tool_calls=[ToolCall(name=n, arguments=a, id=f"call_{i}") for i, (n, a) in enumerate(calls)]
    )
