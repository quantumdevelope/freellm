"""Shared test helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freellm.llm.base import ChatResult, LLMClient, Message, ToolCall, ToolSpec  # noqa: E402


class ScriptedClient(LLMClient):
    """A client that replays a fixed list of responses."""

    name = "scripted"

    def __init__(self, responses: list[ChatResult]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            return ChatResult(content="done", model="scripted")
        return self.responses.pop(0)

    def list_models(self) -> list[str]:
        return ["scripted-model"]


def tool_call(name: str, arguments: dict | None = None, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments or {})
