# Contributing to Mudline

## Getting Started

```bash
# Clone and install
git clone https://github.com/reecelong/mudline.git
cd mudline
pip install -e ".[dev,media]"
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
