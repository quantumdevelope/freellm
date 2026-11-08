"""Shell tool with an allowlist and a workspace sandbox."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from . import Registry, ToolError

DEFAULT_TIMEOUT = 120
MAX_OUTPUT = 20_000


def register(
    registry: Registry,
    root: Path,
    allowed_commands: list[str] | None = None,
    allow_any: bool = False,
) -> None:
    """Register the shell tool on a registry."""
    allowed = set(allowed_commands or [])

    def run_command(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        if not command.strip():
            return "Error: empty command"

        try:
            parts = shlex.split(command, posix=os.name != "nt")
        except ValueError as error:
            return f"Error: cannot parse command: {error}"

        executable = Path(parts[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]

        if not allow_any and executable not in allowed:
            return (
                f"Error: command '{executable}' is not allowed. "
                f"Allowed: {', '.join(sorted(allowed)) or 'none'}. "
                "Change allowed_commands in the config to permit it."
            )

        try:
            completed = subprocess.run(  # noqa: S603 - allowlisted by design
                parts,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        except FileNotFoundError:
            return f"Error: command not found: {parts[0]}"
        except OSError as error:
            return f"Error: {error}"

        output = (completed.stdout or "") + (completed.stderr or "")
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + f"\n[truncated at {MAX_OUTPUT} characters]"
        if not output.strip():
            return f"(no output, exit code {completed.returncode})"
        return output.rstrip()

    registry.register(
        "run_command",
        "Run a shell command in the workspace. Only allowlisted commands are permitted.",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command line to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds"},
            },
            "required": ["command"],
        },
        run_command,
        tags=["shell"],
    )
