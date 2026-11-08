"""OpenAI-compatible backend.

Works with every server that speaks the OpenAI chat-completions API:
Ollama, LM Studio, llama.cpp's ``llama-server``, vLLM, LocalAI, text-generation-webui
and the free tiers of hosted providers.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import ChatResult, LLMClient, LLMError, Message, ToolCall, ToolSpec


def _to_openai_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}

    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]

    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.name:
        payload["name"] = message.name

    return payload


def _parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls or []):
        function = raw.get("function") or {}
        name = function.get("name") or ""
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        calls.append(
            ToolCall(
                id=raw.get("id") or f"call_{index}",
                name=name,
                arguments=arguments if isinstance(arguments, dict) else {"_raw": arguments},
            )
        )
    return calls


class OpenAICompatClient(LLMClient):
    """Talk to an OpenAI-compatible ``/v1`` endpoint."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 180,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    # -- internals ---------------------------------------------------------

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        else:
            # Local servers often ignore the header, but some require it to exist.
            headers["authorization"] = "Bearer freellm-local"

        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as error:  # pragma: no cover - network dependent
            detail = error.read().decode("utf-8", "replace")[:400]
            raise LLMError(f"HTTP {error.code} from {url}: {detail}") from error
        except urllib.error.URLError as error:
            raise LLMError(f"Cannot reach {url}: {error.reason}") from error

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"authorization": f"Bearer {self.api_key or 'freellm-local'}"}
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as error:  # pragma: no cover - network dependent
            raise LLMError(f"HTTP {error.code} from {url}") from error
        except urllib.error.URLError as error:
            raise LLMError(f"Cannot reach {url}: {error.reason}") from error

    # -- public API --------------------------------------------------------

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_message(message) for message in messages],
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if tools:
            body["tools"] = [spec.to_openai() for spec in tools]
            body["tool_choice"] = "auto"

        data = self._post("/chat/completions", body)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"No choices in response: {str(data)[:300]}")

        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}

        return ChatResult(
            content=message.get("content") or "",
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
            model=data.get("model") or self.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            raw=data,
        )

    def list_models(self) -> list[str]:
        try:
            data = self._get("/models")
        except LLMError:
            return []
        models = data.get("data") or []
        names = [str(item.get("id")) for item in models if item.get("id")]
        return sorted(names)
