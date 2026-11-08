"""File system tools."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from . import Registry, ToolError

MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 5_000_000


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _guard_root(root: Path, path: Path) -> None:
    """Keep operations inside the workspace unless the user opted out."""
    try:
        path.relative_to(root)
    except ValueError:
        raise ToolError(f"Path escapes the workspace: {path}")


def register(registry: Registry, root: Path, allow_outside: bool = False) -> None:
    """Register the file tools on a registry."""

    def read_file(path: str, max_bytes: int = MAX_READ_BYTES) -> str:
        target = _resolve(root, path)
        if not allow_outside:
            _guard_root(root, target)
        if not target.exists():
            return f"Error: file not found: {target}"
        if target.is_dir():
            return f"Error: {target} is a directory, use list_dir"
        data = target.read_bytes()[:max_bytes]
        text = data.decode("utf-8", "replace")
        truncated = len(data) < target.stat().st_size
        return f"{text}\n[truncated at {max_bytes} bytes]" if truncated else text

    def write_file(path: str, content: str) -> str:
        target = _resolve(root, path)
        if not allow_outside:
            _guard_root(root, target)
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            return "Error: content too large"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {target}"

    def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        target = _resolve(root, path)
        if not allow_outside:
            _guard_root(root, target)
        if not target.exists():
            return f"Error: file not found: {target}"
        text = target.read_text(encoding="utf-8", errors="replace")
        occurrences = text.count(old_string)
        if occurrences == 0:
            return "Error: old_string not found in file"
        if occurrences > 1 and not replace_all:
            return (
                f"Error: old_string appears {occurrences} times; pass replace_all=true "
                "or include more surrounding context"
            )
        text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        target.write_text(text, encoding="utf-8")
        return f"Edited {target} ({occurrences} replacement(s))"

    def list_dir(path: str = ".") -> str:
        target = _resolve(root, path)
        if not allow_outside:
            _guard_root(root, target)
        if not target.exists():
            return f"Error: directory not found: {target}"
        if not target.is_dir():
            return f"Error: {target} is not a directory"
        entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        lines = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            kind = "d" if entry.is_dir() else "f"
            size = "" if entry.is_dir() else f" {entry.stat().st_size}b"
            lines.append(f"{kind} {entry.name}{size}")
        return "\n".join(lines) or "(empty directory)"

    def find_files(pattern: str = "*", path: str = ".", limit: int = 200) -> str:
        base = _resolve(root, path)
        if not allow_outside:
            _guard_root(root, base)
        if not base.is_dir():
            return f"Error: not a directory: {base}"
        matches: list[str] = []
        for current, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    relative = Path(current, name).relative_to(base)
                    matches.append(relative.as_posix())
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break
        return "\n".join(matches) or "(no matches)"

    def search_text(query: str, path: str = ".", limit: int = 100) -> str:
        base = _resolve(root, path)
        if not allow_outside:
            _guard_root(root, base)
        if not base.is_dir():
            return f"Error: not a directory: {base}"
        hits: list[str] = []
        for current, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
            for name in files:
                candidate = Path(current, name)
                try:
                    if candidate.stat().st_size > MAX_READ_BYTES:
                        continue
                    content = candidate.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for number, line in enumerate(content.splitlines(), start=1):
                    if query in line:
                        hits.append(f"{candidate.relative_to(base)}:{number}: {line.strip()[:200]}")
                        if len(hits) >= limit:
                            return "\n".join(hits)
        return "\n".join(hits) or "(no matches)"

    registry.register(
        "read_file",
        "Read a text file from the workspace. Returns the file contents.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the workspace"},
                "max_bytes": {"type": "integer", "description": "Maximum bytes to read"},
            },
            "required": ["path"],
        },
        read_file,
        tags=["fs"],
    )
    registry.register(
        "write_file",
        "Create or overwrite a text file in the workspace.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        write_file,
        tags=["fs"],
    )
    registry.register(
        "edit_file",
        "Replace an exact string inside an existing file.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        edit_file,
        tags=["fs"],
    )
    registry.register(
        "list_dir",
        "List the entries of a directory.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
        list_dir,
        tags=["fs"],
    )
    registry.register(
        "find_files",
        "Find files by glob pattern inside the workspace.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob, e.g. *.py"},
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
        find_files,
        tags=["fs"],
    )
    registry.register(
        "search_text",
        "Search file contents for a literal string and return matching lines.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        search_text,
        tags=["fs"],
    )
