# freellm

**A free AI coding agent for your terminal.** Runs entirely on your own
machine — on Ollama, llama.cpp, LM Studio, vLLM or any OpenAI-compatible
server. No API keys, no accounts, no usage bills.

```bash
freellm                       # interactive chat
freellm run "fix the failing test"
freellm serve                 # OpenAI-compatible API on :8787
freellm doctor                # check that everything is wired up
```

- 💸 **Truly free** — inference runs locally, so every request costs **$0.00**
- 🧰 **Real tools** — read/write/edit files, run commands, fetch URLs, search a local index
- 🔌 **MCP** — use any Model Context Protocol server (stdio) as extra tools
- 🧩 **Plugins** — drop a Python file in `~/.freellm/plugins` and it joins the agent
- 🧠 **Local RAG** — index your workspace and let the agent search it
- 🌐 **OpenAI-compatible server** — point any client at freellm, still for free
- 🪶 **No runtime dependencies** — the standard library is enough

---

## Why

Every serious coding agent wants an API key. Those keys cost money, leak, and
send your code to someone else's servers. freellm is the opposite: point it at a
model you already run, and it stays free and private forever.

```bash
# one-time setup
ollama serve
ollama pull llama3.2
```

That is the whole setup. No signup, no key, no card.

## Install

```bash
pip install freellm          # from PyPI (once published)
pipx install freellm         # isolated install, recommended
```

From source:

```bash
git clone https://github.com/quantumdevelope/freellm.git
cd freellm
pip install -e .
```

Python 3.10+, no runtime dependencies.

## Configure

Everything is local. `~/.freellm/config.json`:

```json
{
  "provider": "ollama",
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "llama3.2",
  "temperature": 0.7,
  "max_steps": 12,
  "workspace": "/path/to/project",
  "allowed_commands": ["git", "node", "npm", "python", "pytest"],
  "allow_any_command": false
}
```

Environment variables override the file: `FREELLM_MODEL`, `FREELLM_BASE_URL`,
`FREELLM_PROVIDER`, `FREELLM_TEMPERATURE`, `FREELLM_MAX_STEPS`,
`FREELLM_WORKSPACE`, `FREELLM_API_KEY`.

Any OpenAI-compatible server works — freellm detects a `/v1` base URL
automatically:

```bash
freellm --base-url http://127.0.0.1:8080/v1 --model qwen2.5-coder run "explain this repo"
```

## Usage

### Interactive

```console
$ freellm
freellm — free, local-first AI agent
/free    free models   /model <name>   /tools   /mcp   /clear   /help   /exit

› find where the auth token is validated and add a test for it
  → search_text(query='validate_token')
    src/auth/session.py:42: def validate_token(token: str) -> bool:
  → edit_file(path='tests/test_auth.py', old_string='...', new_string='...')
    Edited tests/test_auth.py (1 replacement(s))

Added `test_validate_token_rejects_expired` in tests/test_auth.py.
```

### One-shot

```bash
freellm run "summarise the TODO comments in src/"
freellm run "add type hints to utils.py" --json
```

### Built-in tools

| Tool | What it does |
| --- | --- |
| `read_file` | Read a text file from the workspace |
| `write_file` | Create or overwrite a file |
| `edit_file` | Replace an exact string in a file |
| `list_dir` | List a directory |
| `find_files` | Find files by glob |
| `search_text` | Grep-like content search |
| `run_command` | Run an allowlisted shell command inside the workspace |
| `fetch_url` | Download a URL as plain text |
| `index_workspace` | Index the workspace into the local search index |
| `search_workspace` | Search the local index |

The shell tool is sandboxed to the workspace and only runs allowlisted
commands. Extend the list in `allowed_commands`, or set `allow_any_command` to
`true` if you know what you are doing.

### MCP servers

`~/.freellm/mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] }
  }
}
```

Every MCP tool is exposed to the model with a `mcp_<server>_<tool>` name.

### Plugins

A plugin is one Python file with a `register(registry, context)` function:

```python
# ~/.freellm/plugins/hello.py
def register(registry, context):
    registry.register(
        "say_hello",
        "Greet someone.",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        lambda name: f"Hello, {name}!",
    )
```

Restart freellm and `/tools` will list `say_hello`. Plugins can also be shared
as normal Python packages through the `freellm.plugins` entry point.

### Local RAG

```bash
freellm index                    # index the workspace
freellm                           # then ask: "where is the retry logic?"
```

Embeddings come from the same local model, so indexing stays free and offline.

### OpenAI-compatible server

```bash
freellm serve --port 8787
```

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"hello"}]}'
```

Any tool that speaks the OpenAI API can use it — IDE extensions, scripts, other
agents — and it is still running on your hardware.

## Cost

| | Cost |
| --- | --- |
| Inference | $0.00 (runs on your machine) |
| Telemetry | none, there is no telemetry |
| Accounts | none |

The only cost is the electricity for your own hardware.

## Development

```bash
git clone https://github.com/quantumdevelope/freellm.git
cd freellm
pip install -e ".[dev]"
python -m unittest discover -s tests -t tests
```

The test suite is standard-library only and never touches the network.

## Roadmap

See [ROADMAP.md](ROADMAP.md). Big items: multi-agent orchestration, a plugin
marketplace, a web UI, and a provider-agnostic model router.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests are welcome — especially
tools, providers and plugins.

## License

MIT © freellm contributors
