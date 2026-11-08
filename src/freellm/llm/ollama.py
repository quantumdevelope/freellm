"""Native Ollama backend and model discovery."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import ChatResult, LLMClient, LLMError, Message, ToolCall, ToolSpec


class OllamaClient(LLMClient):
    """Talk to a local Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.2",
        timeout: int = 180,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[: -len("/v1")]
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as error:  # pragma: no cover - network dependent
            detail = error.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"HTTP {error.code} from {url}: {detail}") from error
        except urllib.error.URLError as error:
            raise LLMError(f"Cannot reach {url}: {error.reason}") from error

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as error:  # pragma: no cover - network dependent
            raise LLMError(f"HTTP {error.code} from {url}") from error
        except urllib.error.URLError as error:
            raise LLMError(f"Cannot reach {url}: {error.reason}") from error

    @staticmethod
    def _to_ollama_message(message: Message) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "content": message.content,
            }
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                }
                for call in message.tool_calls
            ]
        return payload

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_ollama_message(message) for message in messages],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
                for spec in tools
            ]

        data = self._post("/api/chat", body)
        message = data.get("message") or {}

        calls: list[ToolCall] = []
        for index, raw in enumerate(message.get("tool_calls") or []):
            function = raw.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{index}",
                    name=function.get("name") or "",
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        return ChatResult(
            content=message.get("content") or "",
            tool_calls=calls,
            model=data.get("model") or self.model,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            raw=data,
        )

    def list_models(self) -> list[str]:
        try:
            data = self._get("/api/tags")
        except LLMError:
            return []
        names = [str(item.get("name")) for item in (data.get("models") or []) if item.get("name")]
        return sorted(names)

    def embed(self, text: str) -> list[float]:
        data = self._post("/api/embeddings", {"model": self.model, "prompt": text})
        embedding = data.get("embedding") or []
        return [float(value) for value in embedding]
