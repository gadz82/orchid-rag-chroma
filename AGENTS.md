# orchid-rag-chroma — AI Context

## What This Package Is

`orchid-rag-chroma` is the ChromaDB vector-backend plugin for the Orchid AI
framework. It provides:

- `ChromaRepository` — implements `OrchidVectorStoreRepository` (read + write + admin)

## Auto-Registration

The package registers itself via Python `importlib.metadata` entry points:

```toml
[project.entry-points."orchid.vector_backends"]
chroma = "orchid_rag_chroma:_register"
```

No manual `register_vector_backend()` calls are needed by integrators.

## Key Files

| File | Purpose |
|------|---------|
| `repository.py` | `ChromaRepository`, `_build_where`, `_sanitize_metadata` |
| `__init__.py` | Entry-point `_register()` callable |

## Testing

Tests require `chromadb` but do **not** require a live ChromaDB server —
all unit tests mock the AsyncHttpClient.

```bash
cd orchid-rag-chroma
pip install -e ".[dev]"
pytest tests/ -x
```

## Common Pitfalls

- ChromaDB **metadata filters are flat** — nested dicts are dropped by
  `_sanitize_metadata()` before writing.
- ChromaDB's `query()` returns **distances**, not scores.  The repository
  converts `1.0 - distance` to a `[0.0, 1.0]` score.
- Embedding dimension mismatches cause silent retrieval failures.
  Switching models requires re-creating ChromaDB collections.
