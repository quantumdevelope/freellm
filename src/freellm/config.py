"""Configuration: file, environment and CLI overrides.

Everything is local. No account, no API key, no telemetry.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "llama3.2"

#: Commands the shell tool may run unless the user opts out.
DEFAULT_ALLOWED_COMMANDS = [
    "git",
    "node",
    "npm",
    "npx",
    "python",
    "python3",
    "pip",
    "pip3",
    "ls",
    "cat",
    "echo",
    "pwd",
    "cd",
    "rg",
    "grep",
    "find",
    "head",
    "tail",
    "wc",
    "make",
    "cargo",
    "go",
    "dotnet",
]


def home_dir() -> Path:
    """Directory holding freellm state (config, databases, plugins)."""
    override = os.environ.get("FREELLM_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".freellm"


def config_path() -> Path:
    return home_dir() / "config.json"


@dataclass
class Config:
    """Runtime configuration."""

    provider: str = "ollama"
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    max_steps: int = 12
    request_timeout: int = 180
    workspace: str = field(default_factory=lambda: str(Path.cwd()))
    allowed_commands: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_COMMANDS))
    allow_any_command: bool = False
    enable_web: bool = True
    enable_mcp: bool = True
    embedding_model: str = "nomic-embed-text"
    system_prompt_extra: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | None = None) -> Path:
        target = path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


_ENV_PREFIX = "FREELLM_"


def _apply_env(config: Config, env: dict[str, str]) -> Config:
    mapping = {
        "PROVIDER": ("provider", str),
        "BASE_URL": ("base_url", str),
        "MODEL": ("model", str),
        "API_KEY": ("api_key", str),
        "TEMPERATURE": ("temperature", float),
        "MAX_TOKENS": ("max_tokens", int),
        "MAX_STEPS": ("max_steps", int),
        "WORKSPACE": ("workspace", str),
        "EMBEDDING_MODEL": ("embedding_model", str),
    }
    for suffix, (attr, caster) in mapping.items():
        raw = env.get(_ENV_PREFIX + suffix)
        if raw is None or raw == "":
            continue
        try:
            setattr(config, attr, caster(raw))
        except (TypeError, ValueError):
            continue
    return config


def load_config(path: Path | None = None, env: dict[str, str] | None = None) -> Config:
    """Load configuration from disk, then apply environment overrides."""
    config = Config()
    target = path or config_path()

    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        for f in fields(Config):
            if f.name in data:
                setattr(config, f.name, data[f.name])

    return _apply_env(config, env if env is not None else dict(os.environ))
