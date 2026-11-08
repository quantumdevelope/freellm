"""Tool registry.

A tool is a JSON-schema description plus a Python callable. Plugins register
their own tools here, exactly like the built-ins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..llm.base import ToolSpec

Handler = Callable[..., str]


@dataclass
class Tool:
    """A callable the model may invoke."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    tags: list[str] = field(default_factory=list)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolError(RuntimeError):
    """Raised when a tool cannot complete its work."""


class Registry:
    """Holds the tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Handler,
        tags: list[str] | None = None,
    ) -> Tool:
        if name in self._tools:
            raise ToolError(f"Tool already registered: {name}")
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            tags=list(tags or []),
        )
        self._tools[name] = tool
        return tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec() for name in sorted(self._tools)]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self.names()) or 'none'}"

        try:
            result = tool.handler(**(arguments or {}))
        except TypeError as error:
            return f"Error: invalid arguments for '{name}': {error}"
        except Exception as error:  # noqa: BLE001 - tools must never crash the agent
            return f"Error: {type(error).__name__}: {error}"

        return result if isinstance(result, str) else str(result)
