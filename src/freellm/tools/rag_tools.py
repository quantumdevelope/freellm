"""RAG tools exposed to the model."""

from __future__ import annotations

from pathlib import Path

from ..rag import RagStore, index_directory, search
from . import Registry


def register(
    registry: Registry,
    store: RagStore,
    embed,
    root: Path,
) -> None:
    """Register index/search tools on a registry."""

    def index_workspace() -> str:
        files, chunks = index_directory(store, root, embed)
        return f"Indexed {files} file(s), {chunks} chunk(s) from {root}"

    def search_workspace(query: str, top_k: int = 5) -> str:
        hits = search(store, query, embed, top_k=top_k)
        if not hits:
            return "No matches. Run index_workspace first."
        return "\n\n".join(
            f"{hit.path}#{hit.index} (score {hit.score:.3f})\n{hit.content[:800]}" for hit in hits
        )

    registry.register(
        "index_workspace",
        "Index the workspace files into the local search index.",
        {"type": "object", "properties": {}},
        index_workspace,
        tags=["rag"],
    )
    registry.register(
        "search_workspace",
        "Search the local index for relevant code or notes.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        search_workspace,
        tags=["rag"],
    )
