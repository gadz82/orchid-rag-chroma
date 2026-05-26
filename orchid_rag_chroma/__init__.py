"""ChromaDB vector-backend for the Orchid AI framework.

Auto-registers via ``importlib.metadata`` entry points — no manual
``register_vector_backend()`` calls needed.
"""

from __future__ import annotations

import logging
from typing import Any

__version__ = "1.0.0"

from .repository import ChromaRepository

__all__ = ["ChromaRepository"]

logger = logging.getLogger(__name__)


def _build_chroma_reader(
    *,
    chroma_client_type: str = "http",
    chroma_host: str = "localhost",
    chroma_port: int = 8000,
    chroma_path: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    **_settings: Any,
) -> ChromaRepository:
    import os as _os

    from orchid_ai.rag.embeddings import build_embeddings, get_embedding_dimension

    if not chroma_path:
        chroma_path = _os.environ.get("CHROMA_PATH")
        if chroma_path:
            chroma_client_type = "persistent"

    embeddings = build_embeddings(embedding_model)
    dimension = get_embedding_dimension(embedding_model)
    return ChromaRepository(
        client_type=chroma_client_type,
        host=chroma_host,
        port=chroma_port,
        path=chroma_path,
        embeddings=embeddings,
        embedding_dimension=dimension,
    )


def _register() -> None:
    """Entry-point callable — invoked by ``orchid_ai.rag.factory`` at import time."""
    try:
        from orchid_ai.rag.factory import register_vector_backend

        register_vector_backend("chroma", _build_chroma_reader)
        logger.debug("[orchid-rag-chroma] Registered vector backend")
    except ImportError:
        logger.debug("[orchid-rag-chroma] Skipping registration (not in this orchid-ai version)")
