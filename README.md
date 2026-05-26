# orchid-rag-chroma

ChromaDB vector backend plugin for the [Orchid AI](https://github.com/gadz82/orchid) framework.

## What it provides

- `ChromaRepository` — implements `OrchidVectorStoreRepository` backed by ChromaDB

## Installation

```bash
pip install orchid-rag-chroma
```

## Usage

Reference `vector_backend: chroma` in your `agents.yaml`:

```yaml
rag:
  vector_backend: chroma
  chroma_client_type: http
  chroma_host: localhost
  chroma_port: 8000
```

Or build it programmatically:

```python
from orchid_rag_chroma import ChromaRepository
from langchain_community.embeddings import OllamaEmbeddings

repo = ChromaRepository(
    client_type="http",
    host="localhost",
    port=8000,
    embeddings=OllamaEmbeddings(model="nomic-embed-text"),
    embedding_dimension=768,
)
```

## Development

```bash
cd orchid-rag-chroma
pip install -e ".[dev]"
pytest tests/ -x
ruff check orchid_rag_chroma/
```

## License

MIT
