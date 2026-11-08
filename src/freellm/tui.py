"""A small terminal UI for the interactive chat."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"


def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return os.environ.get("TERM") != "dumb"


def ensure_utf8_output() -> None:
    """Make sure box-drawing characters survive a cp1251/cp437 console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = (getattr(stream, "encoding", "") or "").lower()
        if encoding in ("utf-8", "utf8"):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            pass


class Console:
    """Tiny colour helper."""

    def __init__(self, enabled: bool | None = None) -> None:
        ensure_utf8_output()
        self.enabled = supports_color() if enabled is None else enabled

    def paint(self, text: str, code: str) -> str:
        if not self.enabled:
            return text
        return f"{code}{text}{RESET}"

    def print(self, text: str = "") -> None:
        print(text)

    def error(self, text: str) -> None:
        print(self.paint(text, RED), file=sys.stderr)


BANNER = """freellm — free, local-first AI agent
/free    free models   /model <name>   /tools   /mcp   /clear   /help   /exit"""


def render_event(console: Console, kind: str, payload: dict[str, Any]) -> None:
    """Print one agent event."""
    if kind == "assistant":
        content = str(payload.get("content") or "").strip()
        if content:
            console.print(console.paint(content, BOLD))
    elif kind == "tool_call":
        name = payload.get("name")
        arguments = payload.get("arguments") or {}
        summary = ", ".join(f"{key}={value!r}" for key, value in list(arguments.items())[:3])
        console.print(console.paint(f"  → {name}({summary})", MAGENTA))
    elif kind == "tool_result":
        output = str(payload.get("output") or "").strip()
        if output:
            first = output.splitlines()[0][:160]
            console.print(console.paint(f"    {first}", DIM))
    elif kind == "error":
        console.error(f"error: {payload.get('message')}")


def run_repl(
    agent,
    console: Console,
    on_command: Callable[[str], str | None] | None = None,
) -> int:
    """Run the interactive loop until the user leaves."""
    console.print(console.paint(BANNER, CYAN))
    console.print(console.paint("Type a task, or /help for commands.\n", DIM))

    while True:
        try:
            line = input(console.paint("› ", GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0

        if not line:
            continue

        if line.startswith("/"):
            if on_command:
                response = on_command(line)
                if response:
                    console.print(response)
            continue

        result = agent.run(line, handler=lambda kind, payload: render_event(console, kind, payload))
        if result.errors:
            for error in result.errors:
                console.error(error)
        elif result.text:
            console.print()
