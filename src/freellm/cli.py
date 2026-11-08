"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config, config_path, load_config
from .runtime import build_runtime
from .server import FreellmServer, ServerConfig
from .tui import Console, run_repl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freellm",
        description="Free, local-first AI coding agent. Runs on Ollama / llama.cpp / LM Studio — no API keys, no cost.",
    )
    parser.add_argument("--version", action="version", version=f"freellm {__version__}")
    parser.add_argument("--model", help="model name (default: from config)")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL")
    parser.add_argument("--provider", help="ollama | openai-compatible")
    parser.add_argument("--workspace", help="workspace directory the agent works in")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-plugins", action="store_true", help="do not load plugins")
    parser.add_argument("--no-mcp", action="store_true", help="do not start MCP servers")
    parser.add_argument("--no-web", action="store_true", help="disable the web tools")

    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="interactive chat (default)")
    chat.add_argument("--system", help="extra system prompt")

    run = sub.add_parser("run", help="run a single task and exit")
    run.add_argument("task", help="what the agent should do")
    run.add_argument("--system", help="extra system prompt")
    run.add_argument("--json", action="store_true", help="machine-readable output")

    serve = sub.add_parser("serve", help="serve an OpenAI-compatible API locally")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)

    sub.add_parser("doctor", help="check the local setup")
    sub.add_parser("models", help="list models available on the backend")
    sub.add_parser("plugins", help="list loaded plugins and their tools")

    index = sub.add_parser("index", help="index the workspace into the local search index")
    index.add_argument("--path", default=".")

    sub.add_parser("config", help="show the current configuration")

    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    if getattr(args, "model", None):
        config.model = args.model
    if getattr(args, "base_url", None):
        config.base_url = args.base_url
    if getattr(args, "provider", None):
        config.provider = args.provider
    if getattr(args, "workspace", None):
        config.workspace = args.workspace
    if getattr(args, "temperature", None) is not None:
        config.temperature = args.temperature
    if getattr(args, "max_steps", None) is not None:
        config.max_steps = args.max_steps
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(), args)
    command = args.command or "chat"
    console = Console()

    if command == "config":
        print(json.dumps(config.to_dict(), indent=2))
        print(f"\nconfig file: {config_path()}", file=sys.stderr)
        return 0

    if command == "doctor":
        return command_doctor(config, console)

    runtime = build_runtime(
        config,
        with_plugins=not args.no_plugins,
        with_mcp=not args.no_mcp,
        with_web=not args.no_web,
    )

    try:
        if command == "chat":
            return command_chat(runtime, args, console)
        if command == "run":
            return command_run(runtime, args, console)
        if command == "serve":
            return command_serve(runtime, args)
        if command == "models":
            return command_models(runtime, console)
        if command == "plugins":
            return command_plugins(runtime, console)
        if command == "index":
            return command_index(runtime, args, console)
    finally:
        runtime.close()

    return 0


def command_chat(runtime, args: argparse.Namespace, console: Console) -> int:
    extra = getattr(args, "system", "") or runtime.config.system_prompt_extra
    runtime.agent.system_prompt += ("\n" + extra) if extra else ""

    def handle_command(line: str) -> str | None:
        command = line.strip()
        if command in ("/help", "/h"):
            return (
                "/free     show models on the backend\n"
                "/model X  switch model\n"
                "/tools    list tools\n"
                "/mcp      list MCP servers\n"
                "/clear    forget the conversation\n"
                "/exit     leave"
            )
        if command == "/tools":
            return "tools: " + ", ".join(runtime.registry.names())
        if command == "/mcp":
            names = [client.config.name or client.config.command for client in runtime.mcp_clients]
            return "mcp: " + (", ".join(names) if names else "none")
        if command == "/clear":
            runtime.agent.reset()
            return "conversation cleared"
        if command == "/free":
            models = runtime.client.list_models()
            return "models: " + (", ".join(models) if models else "none found")
        if command.startswith("/model"):
            parts = command.split(maxsplit=1)
            if len(parts) == 2:
                runtime.client.model = parts[1]
                return f"model: {parts[1]}"
            return "usage: /model <name>"
        if command in ("/exit", "/quit"):
            raise SystemExit(0)
        return f"unknown command: {command}"

    return run_repl(runtime.agent, console, handle_command)


def command_run(runtime, args: argparse.Namespace, console: Console) -> int:
    extra = getattr(args, "system", "") or ""
    if extra:
        runtime.agent.system_prompt += "\n" + extra

    if args.json:
        result = runtime.agent.run(args.task)
        print(
            json.dumps(
                {
                    "task": args.task,
                    "answer": result.text,
                    "steps": result.steps,
                    "tool_calls": result.tool_calls,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cost_usd": result.cost_usd,
                    "errors": result.errors,
                },
                indent=2,
            )
        )
    else:
        result = runtime.agent.run(
            args.task,
            handler=lambda kind, payload: _echo(console, kind, payload),
        )
        if result.errors:
            for error in result.errors:
                console.error(error)
    return 0


def _echo(console: Console, kind: str, payload: dict) -> None:
    from .tui import render_event

    render_event(console, kind, payload)


def command_serve(runtime, args: argparse.Namespace) -> int:
    server = FreellmServer(
        runtime.agent,
        ServerConfig(host=args.host, port=args.port, model=runtime.config.model),
    )
    server.serve_forever()
    return 0


def command_models(runtime, console: Console) -> int:
    models = runtime.client.list_models()
    if not models:
        console.error(
            f"No models found at {runtime.config.base_url}. "
            "Start Ollama (`ollama serve`, `ollama pull llama3.2`) and try again."
        )
        return 1
    for model in models:
        marker = " *" if model == runtime.config.model else ""
        print(f"{model}{marker}")
    return 0


def command_plugins(runtime, console: Console) -> int:
    print("tools:")
    for name in runtime.registry.names():
        print(f"  {name}")
    if runtime.plugin_errors:
        print("\nplugin errors:")
        for error in runtime.plugin_errors:
            print(f"  {error}")
    return 0


def command_index(runtime, args: argparse.Namespace, console: Console) -> int:
    from .rag import index_directory

    if runtime.store is None:
        console.error("indexing is disabled")
        return 1

    root = Path(args.path).expanduser().resolve()

    def embed(text: str) -> list[float]:
        return runtime.client.embed(text)

    files, chunks = index_directory(runtime.store, root, embed)
    print(f"indexed {files} file(s) -> {chunks} chunk(s) from {root}")
    return 0


def command_doctor(config: Config, console: Console) -> int:
    checks: list[tuple[str, bool, str]] = []

    checks.append(("config file", config_path().exists(), str(config_path())))

    try:
        import urllib.request

        with urllib.request.urlopen(f"{config.base_url.rstrip('/')}/models", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
        models = [item.get("id") for item in payload.get("data") or [] if item.get("id")]
        checks.append(("backend reachable", True, f"{config.base_url} ({len(models)} models)"))
        checks.append(
            (
                "configured model",
                config.model in models,
                f"{config.model}" + ("" if config.model in models else " (not in the list)"),
            )
        )
    except Exception as error:  # noqa: BLE001 - doctor reports, never crashes
        checks.append(("backend reachable", False, f"{config.base_url}: {error}"))

    checks.append(("workspace", Path(config.workspace).exists(), config.workspace))
    checks.append(("cost", True, "$0.00 — inference runs locally"))

    for name, ok, detail in checks:
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")

    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
