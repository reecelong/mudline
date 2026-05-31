# Contributing to Mudline

## Getting Started

```bash
# Clone and install (full app: engine + iOS extraction + media)
git clone https://github.com/reecelong/mudline.git
cd mudline
pip install -e ".[dev,ios,media]"
```

## Development

### Running Tests

```bash
pytest
```

### Linting & Formatting

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
lint-imports        # enforces the engine boundary (see below)
```

### Code Style

- Python 3.12+, type hints everywhere, strict mypy
- Google-style docstrings
- Absolute imports only (`from mudline.models.document import Document`)
- Custom exceptions in `mudline.exceptions` — never bare `except:`
- SQLite: context managers, parameterized queries, no string interpolation
- One logger per module: `logger = logging.getLogger(__name__)`

See `CLAUDE.md` for the full conventions list.

## Architecture

The codebase is organized into five layers, each in its own package:

| Package | Responsibility |
|---------|---------------|
| `mudline.foundation` | Backup discovery, manifest resolution, decryption |
| `mudline.extractors` | Domain-specific parsers (messages, photos, contacts, etc.) |
| `mudline.index` | Vector store, structured store, media index, hybrid retrieval |
| `mudline.intelligence` | Query planning, LLM provider, synthesis |
| `mudline.cli` / `mudline.api` | CLI and REST API interfaces |

### Core Contracts

The shared interfaces in `src/mudline/models/` are the integration point between layers. Every extractor produces `Document` objects. Every retrieval goes through the `Retriever` interface. If you're changing these types, ensure all downstream consumers are updated.

### Engine Boundary

A domain-agnostic **engine** — `mudline.models`, `mudline.index`, the LLM
provider abstraction in `mudline.intelligence.llm`, and `mudline.exceptions` —
is reusable by applications that index data sources other than iOS backups. It
must never import the iOS-specific layers (`foundation`, `extractors`) or the
iOS-tuned query orchestration (`intelligence.planner`/`synthesizer`/`tools`/
etc.). This is enforced by an `import-linter` contract (`pyproject.toml`); run
`lint-imports` to check it. The engine installs without the iOS extras:

```bash
pip install "mudline[vertex]"   # engine + Gemini provider, no iOS deps
```

A consumer produces `Document` objects directly and feeds them to
`IngestPipeline.ingest(...)` — implementing the iOS-only `Extractor` protocol is
not required. Source-specific fields go in the free-form `Document.metadata`
dict and are filterable via `Filters.metadata`. Use `DocumentType.TRANSCRIPT`
for generic (non-iOS) text. For example:

```python
from mudline.models import Document, DocumentType, Source

doc = Document(
    type=DocumentType.TRANSCRIPT,
    text=transcript_text,
    source=Source(backup_id=batch_id, domain="transcripts",
                  relative_path=ref, backup_timestamp=ts),
    metadata={"speaker": "...", "external_id": "..."},  # filterable
)
```

## Adding a New Extractor

1. Create `src/mudline/extractors/<type>.py` implementing the `Extractor` protocol from `src/mudline/models/extractor.py`
2. Register it in `src/mudline/extractors/registry.py`
3. Add tests in `tests/extractors/test_<type>.py`
4. Yield `Document` objects from `src/mudline/models/document.py`

## Pull Requests

- One logical change per PR
- All tests must pass
- New code needs tests
- No cross-layer imports except through `mudline.models`
- Run `ruff check` and `mypy` before submitting
