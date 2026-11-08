"""Plugin loading.

A plugin is a Python file (or package) exposing ``register(registry, context)``.
Plugins live in ``~/.freellm/plugins`` or are declared as ``freellm.plugins``
entry points, which makes them shareable as ordinary Python packages.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import home_dir
from .tools import Registry


@dataclass
class PluginContext:
    """What a plugin gets when it is loaded."""

    workspace: Path
    config: Any = None


@dataclass
class LoadedPlugin:
    """A plugin that was imported successfully."""

    name: str
    source: str
    tools: list[str] = field(default_factory=list)
    error: str | None = None


def plugins_dir() -> Path:
    return home_dir() / "plugins"


def _load_file(path: Path, registry: Registry, context: PluginContext) -> LoadedPlugin:
    name = path.stem
    spec = importlib.util.spec_from_file_location(f"freellm_plugin_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    register = getattr(module, "register", None)
    if register is None:
        raise AttributeError(f"Plugin {name} has no register() function")

    before = set(registry.names())
    register(registry, context)
    added = sorted(set(registry.names()) - before)

    return LoadedPlugin(name=name, source=str(path), tools=added)


def _load_entry_points(registry: Registry, context: PluginContext) -> list[LoadedPlugin]:
    from importlib.metadata import entry_points

    loaded: list[LoadedPlugin] = []
    try:
        points = entry_points(group="freellm.plugins")
    except TypeError:  # pragma: no cover - Python < 3.10 shape
        points = entry_points().get("freellm.plugins", [])  # type: ignore[call-arg]

    for point in points:
        before = set(registry.names())
        try:
            plugin = point.load()
            plugin(registry, context)
        except Exception as error:  # noqa: BLE001 - a broken plugin must not break the rest
            loaded.append(LoadedPlugin(name=point.name, source="entry-point", error=str(error)))
            continue
        added = sorted(set(registry.names()) - before)
        loaded.append(LoadedPlugin(name=point.name, source="entry-point", tools=added))

    return loaded


def load_plugins(
    registry: Registry,
    context: PluginContext,
    directory: Path | None = None,
    use_entry_points: bool = True,
) -> list[LoadedPlugin]:
    """Load every plugin that can be found, without failing on broken ones."""
    loaded: list[LoadedPlugin] = []
    base = directory or plugins_dir()

    if base.exists():
        for path in sorted(base.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                loaded.append(_load_file(path, registry, context))
            except Exception as error:  # noqa: BLE001 - keep going
                loaded.append(LoadedPlugin(name=path.stem, source=str(path), error=str(error)))

    if use_entry_points:
        loaded.extend(_load_entry_points(registry, context))

    return loaded
