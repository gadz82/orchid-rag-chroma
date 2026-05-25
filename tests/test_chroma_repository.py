"""Tests for ``ChromaRepository`` against a mocked ``chromadb`` client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from orchid_ai.core.scopes import OrchidRAGScope

from orchid_rag_chroma.repository import ChromaRepository, _build_where, _sanitize_metadata


# ── Helpers ─────────────────────────────────────────────────


def _mock_embeddings() -> MagicMock:
    emb = MagicMock()
    emb.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    return emb


def _mock_http_client() -> MagicMock:
    client = MagicMock()
    collection = MagicMock()

    # get_or_create_collection returns the mock collection
    client.get_or_create_collection = AsyncMock(return_value=collection)

    # collection.query returns a minimal result
    collection.query = AsyncMock(
        return_value={
            "ids": [["doc-1"]],
            "documents": [["hello"]],
            "metadatas": [[{"k": "v"}]],
            "distances": [[0.25]],
        }
    )

    # collection.add / upsert / delete
    collection.add = AsyncMock()
    collection.upsert = AsyncMock()
    collection.delete = AsyncMock()

    return client


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")


def _repo(client: object | None = None) -> ChromaRepository:
    return ChromaRepository(
        client_type="http",
        embeddings=_mock_embeddings(),
        embedding_dimension=768,
        client=client or _mock_http_client(),
    )


# ── Construction ──────────────────────────────────────────────


class TestConstruction:
    def test_missing_driver_raises_import_error(self):
        with patch.dict("sys.modules", {"chromadb": None}):
            with pytest.raises(ImportError, match="pip install orchid-rag-chroma"):
                ChromaRepository(embeddings=MagicMock(), embedding_dimension=768)


# ── Ensure collections ───────────────────────────────────────


class TestEnsureCollections:
    @pytest.mark.asyncio
    async def test_creates_multiple_collections(self):
        client = _mock_http_client()
        repo = _repo(client)
        await repo.ensure_collections(["ns_a", "ns_b"])
        assert client.get_or_create_collection.await_count == 2


# ── Index ────────────────────────────────────────────────────


class TestIndex:
    @pytest.mark.asyncio
    async def test_adds_documents(self):
        client = _mock_http_client()
        repo = _repo(client)
        docs = [Document(page_content="hello", metadata={"k": "v"}, id="d1")]
        await repo.index(docs, "ns")

        coll = client.get_or_create_collection.return_value
        coll.add.assert_awaited_once()
        call = coll.add.await_args
        assert call.kwargs["ids"] == ["d1"]
        assert call.kwargs["documents"] == ["hello"]
        assert call.kwargs["metadatas"] == [{"k": "v"}]

    @pytest.mark.asyncio
    async def test_empty_list_noops(self):
        client = _mock_http_client()
        repo = _repo(client)
        await repo.index([], "ns")
        client.get_or_create_collection.assert_not_awaited()


# ── Upsert ───────────────────────────────────────────────────


class TestUpsert:
    @pytest.mark.asyncio
    async def test_upserts_documents(self):
        client = _mock_http_client()
        repo = _repo(client)
        docs = [Document(page_content="hello", id="d1")]
        await repo.upsert(docs, "ns")

        coll = client.get_or_create_collection.return_value
        coll.upsert.assert_awaited_once()
        assert coll.upsert.await_args.kwargs["ids"] == ["d1"]


# ── Delete ───────────────────────────────────────────────────


class TestDelete:
    @pytest.mark.asyncio
    async def test_deletes_by_ids(self):
        client = _mock_http_client()
        repo = _repo(client)
        await repo.delete(["d1", "d2"], "ns")

        coll = client.get_or_create_collection.return_value
        coll.delete.assert_awaited_once()
        assert coll.delete.await_args.kwargs["ids"] == ["d1", "d2"]

    @pytest.mark.asyncio
    async def test_empty_list_noops(self):
        client = _mock_http_client()
        repo = _repo(client)
        await repo.delete([], "ns")
        client.get_or_create_collection.assert_not_called()


# ── Retrieve ─────────────────────────────────────────────────


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_returns_parsed_results(self):
        client = _mock_http_client()
        repo = _repo(client)
        results = await repo.retrieve(query="hello", namespace="ns", k=3)

        assert len(results) == 1
        assert results[0].document.page_content == "hello"
        assert results[0].document.id == "doc-1"
        assert results[0].document.metadata == {"k": "v"}
        assert results[0].score > 0.5  # 1.0 - 0.25 = 0.75

    @pytest.mark.asyncio
    async def test_empty_results(self):
        client = _mock_http_client()
        coll = client.get_or_create_collection.return_value
        coll.query = AsyncMock(return_value={"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})

        repo = _repo(client)
        results = await repo.retrieve(query="nothing", namespace="ns")
        assert results == []

    @pytest.mark.asyncio
    async def test_passes_scope_filter(self):
        client = _mock_http_client()
        repo = _repo(client)
        await repo.retrieve(query="hello", namespace="ns", scope=_scope())

        coll = client.get_or_create_collection.return_value
        call = coll.query.await_args
        assert call.kwargs["where"] is not None
        assert "$or" in call.kwargs["where"]


# ── _build_where ──────────────────────────────────────────────


class TestBuildWhere:
    def test_none_scope_none_filters(self):
        assert _build_where(None, None) is None

    def test_tenant_only_scope(self):
        scope = OrchidRAGScope(tenant_id="t1")
        where = _build_where(scope, None)
        assert where is not None
        assert "$or" in where
        # Should have shared + tenant + (tenant, scope=user since user_id is empty? no, user_id is "")
        # user_id is "" so the user clause should NOT be added
        assert len(where["$or"]) == 2  # shared + tenant

    def test_full_scope(self):
        scope = OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")
        where = _build_where(scope, None)
        assert where is not None
        clauses = where["$or"]
        assert len(clauses) == 5  # shared, tenant, user, chat_shared, chat_agent

    def test_metadata_scalar_filter(self):
        where = _build_where(None, {"status": "published"})
        assert where == {"status": "published"}

    def test_metadata_match_any_filter(self):
        where = _build_where(None, {"lang": ["en", "fr"]})
        assert where == {"$or": [{"lang": "en"}, {"lang": "fr"}]}

    def test_metadata_not_filter(self):
        where = _build_where(None, {"active": {"not": False}})
        assert where == {"active": {"$ne": False}}

    def test_metadata_range_filter(self):
        where = _build_where(None, {"count": {"gte": 10, "lt": 100}})
        assert where == {"count": {"$gte": 10, "$lt": 100}}

    def test_metadata_contains_filter(self):
        where = _build_where(None, {"tags": {"contains": "alpha"}})
        assert where == {"tags": {"$contains": "alpha"}}

    def test_backend_namespaced_keys_skipped(self):
        where = _build_where(None, {"_qdrant": {"x": 1}, "status": "ok"})
        assert where == {"status": "ok"}

    def test_scope_plus_metadata_combined_with_and(self):
        scope = OrchidRAGScope(tenant_id="t1")
        where = _build_where(scope, {"status": "published"})
        assert "$and" in where
        assert len(where["$and"]) == 2

    def test_empty_filters_returns_none(self):
        assert _build_where(None, {}) is None


# ── _sanitize_metadata ───────────────────────────────────────


class TestSanitizeMetadata:
    def test_removes_none_values(self):
        assert _sanitize_metadata({"a": "ok", "b": None}) == {"a": "ok"}

    def test_keeps_primitives(self):
        assert _sanitize_metadata({"s": "x", "i": 1, "f": 1.0, "b": True}) == {
            "s": "x",
            "i": 1,
            "f": 1.0,
            "b": True,
        }

    def test_drops_nested_dicts(self):
        assert _sanitize_metadata({"a": {"x": 1}, "b": "ok"}) == {"b": "ok"}

    def test_drops_lists(self):
        assert _sanitize_metadata({"a": [1, 2], "b": "ok"}) == {"b": "ok"}
