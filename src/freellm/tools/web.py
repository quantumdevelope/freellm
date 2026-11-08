"""Web tools — free, no API key required."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from html import unescape

from . import Registry

MAX_BYTES = 2_000_000
DEFAULT_TIMEOUT = 30

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def html_to_text(html: str) -> str:
    """Rough plain-text rendering of an HTML document."""
    without_scripts = _SCRIPT_RE.sub(" ", html)
    without_tags = _TAG_RE.sub(" ", without_scripts)
    text = unescape(without_tags)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def register(registry: Registry, timeout: int = DEFAULT_TIMEOUT) -> None:
    """Register web tools on a registry."""

    def fetch_url(url: str, raw: bool = False) -> str:
        if not url.startswith(("http://", "https://")):
            return "Error: only http(s) URLs are supported"

        request = urllib.request.Request(
            url,
            headers={
                "user-agent": "freellm/0.1 (+https://github.com/quantumdevelope/freellm)",
                "accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read(MAX_BYTES)
                content_type = response.headers.get("content-type", "")
        except urllib.error.HTTPError as error:
            return f"Error: HTTP {error.code}"
        except urllib.error.URLError as error:
            return f"Error: cannot reach host: {error.reason}"
        except TimeoutError:
            return f"Error: timeout after {timeout}s"

        text = data.decode("utf-8", "replace")
        if raw or "html" not in content_type.lower():
            return text
        return html_to_text(text)

    registry.register(
        "fetch_url",
        "Download a URL and return its content as plain text.",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http(s) URL"},
                "raw": {"type": "boolean", "description": "Return the raw body instead of text"},
            },
            "required": ["url"],
        },
        fetch_url,
        tags=["web"],
    )
