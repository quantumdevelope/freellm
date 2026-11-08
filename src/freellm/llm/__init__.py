"""LLM backends."""

from .base import ChatResult, LLMClient, LLMError, Message, ToolCall, ToolSpec
from .ollama import OllamaClient
from .openai_compat import OpenAICompatClient

__all__ = [
    "ChatResult",
    "LLMClient",
    "LLMError",
    "Message",
    "OllamaClient",
    "OpenAICompatClient",
    "ToolCall",
    "ToolSpec",
    "build_client",
]


def build_client(config) -> LLMClient:
    """Create the client described by a :class:`freellm.config.Config`.

    A base URL that already points at an OpenAI-compatible ``/v1`` endpoint is
    detected automatically, so ``--base-url http://host:1234/v1`` just works.
    """
    provider = (config.provider or "ollama").lower()
    base_url = (config.base_url or "").rstrip("/")

    if provider in ("ollama", "native") and not base_url.endswith("/v1"):
        return OllamaClient(
            base_url=base_url or config.base_url,
            model=config.model,
            timeout=config.request_timeout,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    return OpenAICompatClient(
        base_url=base_url,
        model=config.model,
        api_key=config.api_key,
        timeout=config.request_timeout,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
