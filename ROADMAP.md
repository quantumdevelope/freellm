# Roadmap

The goal: a free, local-first AI coding agent that is genuinely good to work
with, and an ecosystem around it.

## v0.1 — the agent (current)

- [x] Agent loop with tool calling
- [x] File tools: read, write, edit, list, find, search
- [x] Shell tool with an allowlist and a workspace sandbox
- [x] Web tool (free, no key)
- [x] Ollama and OpenAI-compatible backends
- [x] MCP client (stdio)
- [x] Plugin loader (`~/.freellm/plugins` and entry points)
- [x] Local RAG: chunking, embeddings, SQLite index, cosine search
- [x] CLI: `chat`, `run`, `serve`, `doctor`, `models`, `plugins`, `index`
- [x] OpenAI-compatible HTTP server
- [x] Standard-library test suite, no network in CI

## v0.2 — better daily driver

- [ ] Streaming output in the TUI (tokens as they arrive)
- [ ] Context budgeting: summarise old turns instead of truncating
- [ ] Multi-file edits in one step (patch application)
- [ ] Diff rendering in the terminal
- [ ] Session transcripts on disk, resumable conversations
- [ ] `/undo` for the last tool action
- [ ] Windows, macOS and Linux path handling parity

## v0.3 — ecosystem

- [ ] Plugin marketplace: `freellm plugin install <name>`
- [ ] Published plugin registry with a simple manifest format
- [ ] Skill files: markdown-defined workflows the agent can load on demand
- [ ] Provider router: pick the cheapest/fastest model per task
- [ ] OpenAI-compatible *provider* mode for free hosted tiers, clearly marked

## v0.4 — scale

- [ ] Multi-agent orchestration (parallel subagents with isolated workspaces)
- [ ] Web UI on top of the same runtime
- [ ] Remote/local hybrid: run the agent here, the model there
- [ ] Structured outputs and JSON-schema-constrained generation
- [ ] Cost dashboard: always $0.00, but show tokens and latency

## Ideas, not scheduled

- Terminal recorder that turns a session into a shareable GIF
- Git-aware tools (branch, diff, staged-aware file reads)
- Sandbox escalation for containers and VMs
- Model-free static analysis mode (no LLM at all)
