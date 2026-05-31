# Mudline — AI-Powered Personal Data Layer over iOS Backups

## What This Project Is

Mudline extracts, indexes, and makes queryable all personal data from iOS backups using natural language. Users can ask questions like "what did Sarah text me about the plumber last month?" and get accurate, cited answers.

## Architecture Overview

```
Interface (FastAPI / CLI / MCP)
    ↓ query
Intelligence (Query Planner → LLM → Synthesizer)
    ↓ retrieval
Index (ChromaDB vectors + SQLite FTS5 + CLIP media)
    ↓ ingest
Extraction (Domain-specific parsers per iOS data type)
    ↓ read
Foundation (ManifestResolver → raw iOS backup files)
```

## Core Contracts (DO NOT MODIFY WITHOUT REVIEW)

The shared interfaces in `src/mudline/models/` are the synchronization point between all layers. Every extractor produces `Document` objects. Every retrieval goes through the `Retriever` interface. Changing these types requires updating all downstream consumers.

- `src/mudline/models/document.py` — Document, Source, Attachment
- `src/mudline/models/extractor.py` — Extractor protocol
- `src/mudline/models/retriever.py` — Retriever interface, Filters, Result

## Package Ownership

| Package | Responsibility |
|---------|---------------|
| `mudline.foundation` | Backup discovery, ManifestResolver, encryption |
| `mudline.extractors` | Domain-specific parsers (messages, photos, contacts, etc.) |
| `mudline.index` | Ingest pipeline, vector store, structured store, hybrid retriever |
| `mudline.intelligence` | Query planner, synthesizer, LLM provider, conversation memory |
| `mudline.cli`, `mudline.api` | CLI and REST API interfaces |

## Code Conventions

- **Python 3.12+**, type hints everywhere, strict mypy
- **Formatter/linter**: ruff (format + lint)
- **Testing**: pytest, all tests in `tests/` mirroring `src/` structure
- **Imports**: absolute imports only (`from mudline.models.document import Document`)
- **Docstrings**: Google style
- **Error handling**: custom exceptions in `mudline.exceptions`, never bare `except:`
- **SQLite**: use context managers, parameterized queries, no string interpolation
- **Logging**: stdlib `logging`, one logger per module (`logger = logging.getLogger(__name__)`)
- **No global state**: pass dependencies explicitly, use dataclasses for config

## Dependencies

Core: `chromadb`, `sentence-transformers`, `plistlib` (stdlib), `sqlite3` (stdlib), `rich`, `textual`, `fastapi`, `uvicorn`

Extraction: `iOSbackup` (encrypted backup support), `protobuf` (Notes decoding)

Intelligence: `anthropic` (Claude API), `openai` (Ollama-compatible), `httpx`

Media: `Pillow`, `open-clip-torch` (CLIP embeddings)

Dev: `ruff`, `mypy`, `pytest`, `pytest-asyncio`

## Key Technical Notes

- iOS backup timestamps use **Cocoa epoch** (seconds since 2001-01-01). Convert with: `datetime(2001, 1, 1) + timedelta(seconds=cocoa_timestamp)`
- Backup files are stored in **256 two-character subdirectories** named by SHA-1 hash prefix
- `Manifest.db` maps `(domain, relativePath) → fileID` (the SHA-1 hash)
- Encrypted backups require keybag parsing from `Manifest.plist` — use `iOSbackup` library
- Notes use **protobuf-encoded rich text** in the `ZICCLOUDSYNCINGOBJECT` column — not plain text
- The LLM provider abstraction must support both cloud APIs (Anthropic, OpenAI-compat) and local Ollama

## Contributing

See `CONTRIBUTING.md` for development setup, code style, and PR guidelines. Key rule: no cross-layer imports except through `mudline.models`.
