"""Local RAG: chunk files, embed them, store them in SQLite, search by cosine.

Embeddings come from the local model, so the whole pipeline stays free and
offline. The store is a plain SQLite file under ``~/.freellm``.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
DEFAULT_EXTENSIONS = [
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".rs",
    ".go",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".java",
    ".sh",
    ".html",
    ".css",
]

Embedder = Callable[[str], list[float]]


@dataclass
class Chunk:
    """One indexed fragment of a file."""

    path: str
    index: int
    content: str
    vector: list[float]


@dataclass
class SearchHit:
    """A chunk that matched a query."""

    path: str
    index: int
    content: str
    score: float


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries when possible."""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + size, length)
        if end < length:
            window = text[start:end]
            boundary = max(window.rfind("\n\n"), window.rfind(". "))
            if boundary > size // 3:
                end = start + boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


class RagStore:
    """SQLite-backed chunk store."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                vector TEXT NOT NULL,
                mtime REAL NOT NULL DEFAULT 0
            )
            """
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
        self.connection.commit()

    def clear(self, path: str | None = None) -> None:
        if path is None:
            self.connection.execute("DELETE FROM chunks")
        else:
            self.connection.execute("DELETE FROM chunks WHERE path = ?", (path,))
        self.connection.commit()

    def upsert(self, path: str, chunks: Iterable[Chunk], mtime: float = 0.0) -> int:
        self.connection.execute("DELETE FROM chunks WHERE path = ?", (path,))
        rows = [(path, chunk.index, chunk.content, _dump(chunk.vector), mtime) for chunk in chunks]
        self.connection.executemany(
            "INSERT INTO chunks (path, chunk_index, content, vector, mtime) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.connection.commit()
        return len(rows)

    def all_chunks(self) -> list[tuple[str, int, str, list[float]]]:
        rows = self.connection.execute("SELECT path, chunk_index, content, vector FROM chunks").fetchall()
        return [(row[0], row[1], row[2], _load(row[3])) for row in rows]

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for path, index, content, vector in self.all_chunks():
            score = cosine(query_vector, vector)
            hits.append(SearchHit(path=path, index=index, content=content, score=score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def close(self) -> None:
        self.connection.close()


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two vectors (0.0 when either is empty)."""
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _dump(vector: list[float]) -> str:
    return ",".join(f"{value:.6f}" for value in vector)


def _load(raw: str) -> list[float]:
    if not raw:
        return []
    try:
        return [float(part) for part in raw.split(",")]
    except ValueError:
        return []


def iter_source_files(root: Path, extensions: list[str] | None = None) -> list[Path]:
    """Find indexable files under a root, skipping noisy directories."""
    allowed = {ext.lower() for ext in (extensions or DEFAULT_EXTENSIONS)}
    skip_dirs = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".next",
        ".cache",
    }

    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in allowed:
            found.append(path)
    return found


def index_directory(
    store: RagStore,
    root: Path,
    embed: Embedder,
    extensions: list[str] | None = None,
) -> tuple[int, int]:
    """Index every source file under ``root``.

    Returns ``(files, chunks)``.
    """
    files = iter_source_files(root, extensions)
    total_chunks = 0

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue

        relative = str(path.relative_to(root)).replace("\\", "/")
        pieces = chunk_text(text)
        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            try:
                vector = embed(piece)
            except Exception:  # noqa: BLE001 - a failed embedding must not abort the run
                vector = []
            chunks.append(Chunk(path=relative, index=index, content=piece, vector=vector))

        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        total_chunks += store.upsert(relative, chunks, mtime)

    return len(files), total_chunks


def search(store: RagStore, query: str, embed: Embedder, top_k: int = 5) -> list[SearchHit]:
    """Search the index for the chunks closest to the query."""
    vector = embed(query)
    return store.search(vector, top_k=top_k)
