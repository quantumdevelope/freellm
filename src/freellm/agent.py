"""The agent loop: model -> tool calls -> results -> model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .llm.base import ChatResult, LLMClient, LLMError, Message, ToolCall
from .tools import Registry

EventHandler = Callable[[str, dict[str, Any]], None]

SYSTEM_PROMPT = """You are freellm, a coding agent that runs entirely on the user's own machine.

Runtime facts:
- Inference is local. Every request costs $0.00 and no API key is involved.
- The workspace directory is {workspace}.
- You can call tools. Prefer using a tool over guessing: read a file before editing it,
  list a directory before creating files in it, and search before writing.
- Work in small, verifiable steps: change one thing, then check the result.
- When a tool reports an error, read the message and fix the cause rather than retrying blindly.
- Answer in the user's language, keep it concise, and show the paths you touched.
"""


@dataclass
class AgentResult:
    """Outcome of one agent run."""

    text: str = ""
    steps: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    messages: list[Message] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Agent:
    """Drives a conversation with tool use."""

    def __init__(
        self,
        client: LLMClient,
        registry: Registry,
        workspace: Path | str = ".",
        max_steps: int = 12,
        system_prompt: str | None = None,
        extra_system_prompt: str = "",
    ) -> None:
        self.client = client
        self.registry = registry
        self.workspace = Path(workspace)
        self.max_steps = max_steps
        self.messages: list[Message] = []
        self.system_prompt = system_prompt or SYSTEM_PROMPT.format(workspace=self.workspace)
        if extra_system_prompt.strip():
            self.system_prompt += "\n" + extra_system_prompt.strip()

    # -- helpers -----------------------------------------------------------

    def reset(self) -> None:
        self.messages = []

    def _emit(self, handler: EventHandler | None, kind: str, payload: dict[str, Any]) -> None:
        if handler:
            handler(kind, payload)

    def _tool_message(self, call: ToolCall, output: str) -> Message:
        return Message(role="tool", content=output, tool_call_id=call.id, name=call.name)

    # -- main loop ---------------------------------------------------------

    def run(self, task: str, handler: EventHandler | None = None) -> AgentResult:
        """Send a task to the model and resolve every tool call it makes."""
        conversation: list[Message] = list(self.messages)
        conversation.append(Message(role="user", content=task))

        result = AgentResult()
        specs = self.registry.specs()

        for step in range(self.max_steps):
            result.steps = step + 1

            try:
                response: ChatResult = self.client.chat(conversation, specs)
            except LLMError as error:
                result.errors.append(str(error))
                self._emit(handler, "error", {"message": str(error)})
                result.messages = conversation
                return result

            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens
            result.cost_usd += response.cost_usd

            conversation.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            self._emit(handler, "assistant", {"content": response.content})

            if not response.tool_calls:
                result.text = response.content
                break

            for call in response.tool_calls:
                result.tool_calls += 1
                self._emit(handler, "tool_call", {"name": call.name, "arguments": call.arguments})

                output = self.registry.call(call.name, call.arguments)
                conversation.append(self._tool_message(call, output))
                self._emit(handler, "tool_result", {"name": call.name, "output": output})
        else:
            result.errors.append(
                f"Stopped after {self.max_steps} steps without a final answer."
            )

        self.messages = conversation
        result.messages = conversation
        return result
