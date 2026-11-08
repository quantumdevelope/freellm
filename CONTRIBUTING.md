# Contributing to freellm

Thanks for helping make a free, local-first coding agent.

## Ground rules

1. **Zero runtime dependencies.** The standard library only. Anything that can be
   implemented in ~200 lines of Python goes into the package instead of a
   dependency.
2. **Local and private by default.** No telemetry, no phone-home, no usage
   reporting. If a feature needs the network, it must be opt-in.
3. **Free stays free.** Do not add features that require a paid API key as the
   only path to work.
4. **The user's machine is the sandbox.** Tools stay inside the workspace and
   the shell tool keeps its allowlist.

## Getting set up

```bash
git clone https://github.com/quantumdevelope/freellm.git
cd freellm
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m unittest discover -s tests -t tests
```

The suite must pass without network access — use the fake backends in
`scripts/` and the scripted client in `tests/helpers.py`.

## Adding a tool

Tools live in `src/freellm/tools/`. A tool is a JSON-schema description plus a
callable:

```python
registry.register(
    "my_tool",
    "What it does, in one line.",
    {"type": "object", "properties": {"arg": {"type": "string"}}, "required": ["arg"]},
    my_handler,
    tags=["custom"],
)
```

Rules:

- The handler returns a string and never raises — return an `Error: ...` string
  instead, so the agent can recover.
- Arguments come straight from the model, so validate them.
- Never escape the workspace unless the user explicitly allowed it.

## Adding a provider

Implement `LLMClient` in `src/freellm/llm/` with `chat()` and `list_models()`,
then wire it into `build_client()` in `src/freellm/llm/__init__.py`.

## Pull requests

- One focused change per PR.
- Add tests for new behaviour.
- Keep the public API stable; breaking changes need a version bump.
- Describe what you tested in the PR body.

## Reporting bugs

Open an issue with the output of `freellm doctor`, your OS and Python version,
and the exact command that misbehaved. A minimal reproduction is worth a
thousand words.
