"""Minimal MCP client (stdio transport, JSON-RPC 2.0).

Supports the parts of the Model Context Protocol that matter for a coding
agent: listing the tools a server exposes and calling them.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools import Registry


class McpError(RuntimeError):
    """Raised when an MCP server misbehaves."""


@dataclass
class McpServerConfig:
    """How to start one MCP server."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    name: str = ""


class McpClient:
    """A connected MCP server."""

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self._next_id = 0
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        if self.process is not None:
            return

        env = {**dict(), **self.config.env}
        try:
            self.process = subprocess.Popen(  # noqa: S603 - command comes from user config
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env or None,
            )
        except OSError as error:
            raise McpError(f"Cannot start MCP server '{self.config.command}': {error}") from error

        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "freellm", "version": "0.1.0"},
            },
        )

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.stdout:
                self.process.stdout.close()
            self.process.wait(timeout=5)
        except Exception:  # noqa: BLE001 - best effort shutdown
            self.process.kill()
        finally:
            self.process = None

    def __enter__(self) -> "McpClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- protocol ----------------------------------------------------------

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the matching response."""
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise McpError("MCP server is not connected")

        with self._lock:
            self._next_id += 1
            request_id = self._next_id

            payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params is not None:
                payload["params"] = params

            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()

            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise McpError(f"MCP server closed the connection during {method}")
                line = line.strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if message.get("id") != request_id:
                    continue  # notification or response to another request

                if "error" in message:
                    error = message["error"]
                    raise McpError(f"{method} failed: {error.get('message', error)}")

                return message.get("result") or {}

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification (no response expected)."""
        if self.process is None or self.process.stdin is None:
            return
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    # -- tools -------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        result = self.request("tools/list")
        return list(result.get("tools") or [])

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}})

        if result.get("isError"):
            return f"Error: MCP tool '{name}' failed"

        parts: list[str] = []
        for item in result.get("content") or []:
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "resource":
                resource = item.get("resource") or {}
                parts.append(str(resource.get("text") or resource.get("uri") or ""))
        return "\n".join(part for part in parts if part)

    def register_into(self, registry: Registry, prefix: str = "") -> list[str]:
        """Expose every MCP tool as a local tool."""
        registered: list[str] = []
        for tool in self.list_tools():
            name = f"{prefix}{tool.get('name', 'tool')}"
            description = tool.get("description") or ""
            parameters = tool.get("inputSchema") or {"type": "object", "properties": {}}

            def handler(**kwargs: Any) -> str:
                return self.call_tool(name.removeprefix(prefix), kwargs)

            registry.register(
                name,
                description,
                parameters,
                handler,
                tags=["mcp"],
            )
            registered.append(name)
        return registered


def load_servers(path: Path) -> list[McpServerConfig]:
    """Read MCP server definitions from a JSON file.

    The file looks like::

        {"mcpServers": {"filesystem": {"command": "npx", "args": ["-y", "server"]}}}
    """
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    servers = data.get("mcpServers") or data.get("servers") or {}
    configs: list[McpServerConfig] = []
    for name, spec in servers.items():
        if not isinstance(spec, dict) or not spec.get("command"):
            continue
        configs.append(
            McpServerConfig(
                command=str(spec["command"]),
                args=[str(arg) for arg in spec.get("args") or []],
                env={str(k): str(v) for k, v in (spec.get("env") or {}).items()},
                name=name,
            )
        )
    return configs
