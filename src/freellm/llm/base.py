"""Shared LLM types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One message in the conversation."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolSpec:
    """JSON-schema description of a tool the model may call."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ChatResult:
    """Model response."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        """Local inference is free — kept for transparency in the UI."""
        return 0.0


class LLMError(RuntimeError):
    """Raised when the model backend cannot be reached or refuses a request."""


class LLMClient:
    """Interface every backend implements."""

    name = "llm"

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> ChatResult:  # pragma: no cover - interface
        raise NotImplementedError

    def list_models(self) -> list[str]:  # pragma: no cover - interface
        return []

    def embed(self, text: str) -> list[float]:  # pragma: no cover - interface
        raise LLMError(f"{self.name} does not support embeddings")
