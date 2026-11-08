"""Assemble the runtime: config -> client -> tools -> agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .agent import Agent
from .config import Config, home_dir
from .llm.base import LLMClient
from .llm import build_client
from .mcp import McpClient, load_servers
from .plugins import PluginContext, load_plugins
from .rag import RagStore
from .tools import Registry
from .tools import fs as fs_tools
from .tools import rag_tools, shell as shell_tools, web as web_tools


@dataclass
class Runtime:
    """Everything a session needs."""

    config: Config
    client: LLMClient
    registry: Registry
    agent: Agent
    store: RagStore | None = None
    mcp_clients: list[McpClient] = field(default_factory=list)
    plugin_errors: list[str] = field(default_factory=list)

    def close(self) -> None:
        for client in self.mcp_clients:
            client.close()
        if self.store:
            self.store.close()


def build_runtime(
    config: Config,
    *,
    with_plugins: bool = True,
    with_mcp: bool = True,
    with_web: bool = True,
    with_rag: bool = True,
) -> Runtime:
    """Create a ready-to-use runtime."""
    workspace = Path(config.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    client = build_client(config)
    registry = Registry()

    fs_tools.register(registry, workspace)
    shell_tools.register(
        registry,
        workspace,
        allowed_commands=config.allowed_commands,
        allow_any=config.allow_any_command,
    )
    if with_web and config.enable_web:
        web_tools.register(registry)

    store: RagStore | None = None
    if with_rag:
        store = RagStore(home_dir() / "index.db")

        def embed(text: str) -> list[float]:
            return client.embed(text)

        rag_tools.register(registry, store, embed, workspace)

    mcp_clients: list[McpClient] = []
    if with_mcp and config.enable_mcp:
        for server_config in load_servers(home_dir() / "mcp.json"):
            client_mcp = McpClient(server_config)
            try:
                client_mcp.connect()
                client_mcp.register_into(registry, prefix=f"mcp_{server_config.name}_" if server_config.name else "")
                mcp_clients.append(client_mcp)
            except Exception as error:  # noqa: BLE001 - a broken server must not block the rest
                client_mcp.close()

    plugin_errors: list[str] = []
    if with_plugins:
        context = PluginContext(workspace=workspace, config=config)
        for plugin in load_plugins(registry, context):
            if plugin.error:
                plugin_errors.append(f"{plugin.name}: {plugin.error}")

    agent = Agent(
        client=client,
        registry=registry,
        workspace=workspace,
        max_steps=config.max_steps,
        extra_system_prompt=config.system_prompt_extra,
    )

    return Runtime(
        config=config,
        client=client,
        registry=registry,
        agent=agent,
        store=store,
        mcp_clients=mcp_clients,
        plugin_errors=plugin_errors,
    )
