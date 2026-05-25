"""
ChromaDB-backed :class:`OrchidVectorStoreRepository`.

Behind the ``chromadb`` package (``pip install chromadb``).
Collections are created lazily on first write; the admin-facing
:meth:`ensure_collections` guarantees they exist before ingestion.

Compared to Qdrant, ChromaDB embeds on the collection side (an
``embedding_function`` wired at collection creation), so the builder
pre-computes embeddings via the LangChain :class:`Embeddings` model
and passes them inline — no per-collection embedding-function handoff.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from orchid_ai.core.repository import OrchidSearchResult, OrchidVectorStoreRepository
from orchid_ai.core.scopes import OrchidRAGScope


class ChromaRepository(OrchidVectorStoreRepository):
    """ChromaDB-backed vector store.

    Parameters
    ----------
    client_type:
        ``"http"``, ``"persistent"``, or ``"ephemeral"`` (default
        ``"http"``).
    host:
        ChromaDB HTTP host (only used when ``client_type="http"``).
    port:
        ChromaDB HTTP port (only used when ``client_type="http"``).
    path:
        Persistence directory (only used when ``client_type="persistent"``).
    embeddings:
        A LangChain :class:`Embeddings` instance.  Embeddings are
        pre-computed and passed inline to ChromaDB — no per-collection
        embedding function is registered.
    embedding_dimension:
        Dimensionality of the embedding vectors.
    """

    def __init__(
        self,
        *,
        client_type: str = "http",
        host: str = "localhost",
        port: int = 8000,
        path: str | None = None,
        embeddings: Embeddings,
        embedding_dimension: int,
        client: object | None = None,
    ) -> None:
        try:
            import chromadb  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "chromadb is not installed. Run `pip install orchid-rag-chroma` to use ChromaRepository."
            ) from exc

        self._client_type = client_type
        self._host = host
        self._port = port
        self._path = path
        self._embeddings = embeddings
        self._embedding_dimension = embedding_dimension

        self._client = client
        self._collections: dict[str, object] = {}
        self._collection_names: set[str] = set()

    async def _get_client(self):
        if self._client is not None:
            return self._client

        import chromadb

        if self._client_type == "http":
            self._client = await chromadb.AsyncHttpClient(
                host=self._host,
                port=self._port,
            )
        elif self._client_type == "persistent":
            self._client = chromadb.PersistentClient(path=self._path or "./chroma_data")
        else:
            self._client = chromadb.EphemeralClient()
        return self._client

    async def _get_collection(self, namespace: str):
        """Return (or lazily create) a collection for *namespace*."""
        if namespace in self._collections:
            return self._collections[namespace]

        client = await self._get_client()
        # ChromaDB's get_or_create_collection is sync for PersistentClient
        # and async for AsyncHttpClient — we handle both.
        get_or_create = getattr(client, "get_or_create_collection", None)
        if get_or_create is None:
            raise RuntimeError("ChromaDB client does not support get_or_create_collection")

        if self._client_type == "http":
            collection = await get_or_create(
                name=namespace,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            collection = get_or_create(
                name=namespace,
                metadata={"hnsw:space": "cosine"},
            )
        self._collections[namespace] = collection
        self._collection_names.add(namespace)
        return collection

    # ── Admin ──────────────────────────────────────────────────

    async def ensure_collections(self, namespaces: list[str]) -> None:
        """Pre-create collections for every namespace."""
        for ns in namespaces:
            await self._get_collection(ns)

    # ── Reader ────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        namespace: str,
        k: int = 5,
        scope: OrchidRAGScope | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[OrchidSearchResult]:
        collection = await self._get_collection(namespace)

        # Pre-compute the query embedding.
        query_embedding = await self._embeddings.aembed_query(query)

        where_filter = _build_where(scope, metadata_filters) if (scope or metadata_filters) else None

        query_fn = getattr(collection, "query", None)
        if self._client_type == "http":
            results = await query_fn(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
            )
        else:
            results = query_fn(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
            )

        return _parse_results(results)

    # ── Writer ────────────────────────────────────────────────

    async def index(
        self,
        documents: list[Document],
        namespace: str,
    ) -> None:
        await self._add_docs("add", documents, namespace)

    async def upsert(
        self,
        documents: list[Document],
        namespace: str,
    ) -> None:
        await self._add_docs("upsert", documents, namespace)

    async def delete(
        self,
        document_ids: list[str],
        namespace: str,
    ) -> None:
        if not document_ids:
            return
        collection = await self._get_collection(namespace)
        del_fn = getattr(collection, "delete", None)
        if self._client_type == "http":
            await del_fn(ids=document_ids)
        else:
            del_fn(ids=document_ids)

    async def _add_docs(self, method: str, documents: list[Document], namespace: str) -> None:
        if not documents:
            return
        collection = await self._get_collection(namespace)

        ids = [doc.id for doc in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [_sanitize_metadata(doc.metadata) for doc in documents]

        embeddings_list = await self._embeddings.aembed_documents(texts)

        fn = getattr(collection, method)
        kwargs = {
            "ids": ids,
            "documents": texts,
            "metadatas": metadatas,
            "embeddings": embeddings_list,
        }
        if self._client_type == "http":
            await fn(**kwargs)
        else:
            fn(**kwargs)


# ── Helpers ─────────────────────────────────────────────────


def _build_where(
    scope: OrchidRAGScope | None,
    metadata_filters: dict[str, object] | None,
) -> dict[str, Any] | None:
    """Translate Orchid scope + metadata filters into a ChromaDB ``where`` dict.

    ChromaDB ``where`` supports a subset of operators:

    * ``{"key": "value"}`` — exact match
    * ``{"key": {"$eq": "value"}}`` — explicit equals
    * ``{"key": {"$ne": "value"}}`` — not equals
    * ``{"$and": [...]}`` / ``{"$or": [...]}`` — logical combinators

    We translate scope into an ``$or`` of {tenant_id, scope} conditions
    (matching the Qdrant filter behaviour), then layer metadata filters
    on top.
    """
    clauses: list[dict[str, Any]] = []

    if scope is not None:
        scope_clauses: list[dict[str, Any]] = []

        # __shared__ visibility
        scope_clauses.append({"tenant_id": "__shared__"})

        # tenant-level
        scope_clauses.append({"$and": [{"tenant_id": scope.tenant_id}, {"scope": "tenant"}]})

        if scope.user_id:
            scope_clauses.append(
                {"$and": [{"tenant_id": scope.tenant_id}, {"user_id": scope.user_id}, {"scope": "user"}]}
            )

        if scope.chat_id:
            scope_clauses.append(
                {
                    "$and": [
                        {"tenant_id": scope.tenant_id},
                        {"user_id": scope.user_id},
                        {"chat_id": scope.chat_id},
                        {"scope": "chat_shared"},
                    ]
                }
            )

        if scope.chat_id and scope.agent_id:
            scope_clauses.append(
                {
                    "$and": [
                        {"tenant_id": scope.tenant_id},
                        {"user_id": scope.user_id},
                        {"chat_id": scope.chat_id},
                        {"agent_id": scope.agent_id},
                        {"scope": "chat_agent"},
                    ]
                }
            )

        if len(scope_clauses) == 1:
            clauses.append(scope_clauses[0])
        else:
            clauses.append({"$or": scope_clauses})

    if metadata_filters:
        for key, value in metadata_filters.items():
            if key.startswith("_"):
                continue  # skip backend-namespaced keys
            clauses.append(_translate_filter_clause(key, value))

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _translate_filter_clause(key: str, value: object) -> dict[str, Any]:
    """Translate a single metadata filter entry into a ChromaDB ``where`` clause."""
    if isinstance(value, list):
        # match-any → $or of $eq
        return {"$or": [{key: v} for v in value]}
    if isinstance(value, dict):
        f = dict(value)
        if "not" in f:
            return {key: {"$ne": f["not"]}}
        if "contains" in f:
            # ChromaDB supports $contains for array-type metadata fields.
            return {key: {"$contains": f["contains"]}}
        # range — translate gte/lte/gt/lt into ChromaDB operators
        chroma_ops: dict[str, Any] = {}
        for op_name, chroma_op in [
            ("gte", "$gte"),
            ("lte", "$lte"),
            ("gt", "$gt"),
            ("lt", "$lt"),
        ]:
            if op_name in f:
                chroma_ops[chroma_op] = f[op_name]
        if chroma_ops:
            return {key: chroma_ops}
        raise ValueError(f"Unknown metadata filter operator in {key!r}: {set(f.keys())}")
    # scalar exact match
    return {key: value}


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB metadata must be flat str/int/float/bool — no None, no nested dicts."""
    clean: dict[str, Any] = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
    return clean


def _parse_results(results: dict) -> list[OrchidSearchResult]:
    """Unpack a ChromaDB ``query()`` response into :class:`OrchidSearchResult` items."""
    ids_outer = results.get("ids")
    if not ids_outer or not ids_outer[0]:
        return []

    ids = ids_outer[0]
    docs = results.get("documents", [[None] * len(ids)])[0]
    mds = results.get("metadatas", [[None] * len(ids)])[0]
    distances = results.get("distances", [[1.0] * len(ids)])[0]

    out: list[OrchidSearchResult] = []
    for i in range(len(ids)):
        doc = Document(
            page_content=docs[i] or "",
            metadata=mds[i] or {},
            id=ids[i],
        )
        # ChromaDB returns *distance* (lower = better for cosine); convert to score.
        d = distances[i] if distances[i] is not None else 1.0
        score = max(0.0, min(1.0, 1.0 - d))
        out.append(OrchidSearchResult(document=doc, score=score))
    return out
