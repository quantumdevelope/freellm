"""An OpenAI-compatible HTTP server in front of the local agent.

Useful for plugging freellm into any tool that speaks the OpenAI API, while
the inference itself stays on your machine and free.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agent import Agent
from .llm.base import Message


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    model: str = "local-model"


def _messages_from_payload(payload: dict[str, Any]) -> list[Message]:
    messages: list[Message] = []
    for item in payload.get("messages") or []:
        role = item.get("role") or "user"
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        messages.append(
            Message(
                role=role,  # type: ignore[arg-type]
                content=str(item.get("content") or ""),
                tool_call_id=item.get("tool_call_id"),
                name=item.get("name"),
            )
        )
    return messages


class _Handler(BaseHTTPRequestHandler):
    server_version = "freellm/0.1"

    # -- helpers -----------------------------------------------------------

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - quieter logs
        return

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") in ("/healthz", "/health"):
            self._send_json(200, {"status": "ok", "cost_usd": 0.0, "free": True})
            return

        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_name,  # type: ignore[attr-defined]
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return

        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": {"message": "not found"}})
            return

        length = int(self.headers.get("content-length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid JSON body"}})
            return

        messages = _messages_from_payload(payload)
        if not messages:
            self._send_json(400, {"error": {"message": "messages is required"}})
            return

        agent: Agent = self.server.agent  # type: ignore[attr-defined]
        agent.messages = [message for message in messages if message.role in ("system", "assistant", "user")]
        agent.messages = [message for message in agent.messages if message.role != "system"]

        last_user = next((message for message in reversed(messages) if message.role == "user"), None)
        task = last_user.content if last_user else ""

        result = agent.run(task)

        self._send_json(
            200,
            {
                "id": f"chatcmpl-{abs(hash(task)) % 10**12}",
                "object": "chat.completion",
                "created": 0,
                "model": self.server.model_name,  # type: ignore[attr-defined]
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop" if not result.errors else "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
                "freellm": {
                    "cost_usd": result.cost_usd,
                    "steps": result.steps,
                    "tool_calls": result.tool_calls,
                },
            },
        )


class FreellmServer:
    """Runs the agent behind an OpenAI-compatible HTTP API."""

    def __init__(self, agent: Agent, config: ServerConfig | None = None) -> None:
        self.agent = agent
        self.config = config or ServerConfig()
        self._server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        return f"http://{self.config.host}:{self.server_port}"

    @property
    def server_port(self) -> int:
        if self._server is None:
            return self.config.port
        return self._server.server_address[1]

    def serve_forever(self) -> None:
        self._server = ThreadingHTTPServer((self.config.host, self.config.port), _Handler)
        self._server.agent = self.agent  # type: ignore[attr-defined]
        self._server.model_name = self.config.model  # type: ignore[attr-defined]
        print(f"freellm server on {self.url} (press Ctrl+C to stop)")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover - interactive
            pass
        finally:
            self._server.server_close()

    def serve_in_thread(self) -> threading.Thread:
        """Start the server in a background thread (used by tests)."""
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread
