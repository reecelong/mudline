# Mudline

> *Named for the boundary where Charleston's tidal rivers meet the pluff mud — where everything settles, layers, and stays.*

AI-powered natural language access to your iOS backup data — messages, photos, contacts, notes, calendar, call history, and more.

## What It Does

Ask questions about your own data in plain English:

```
$ mudline ask "what did Sarah text me about the plumber last month?"

Based on your messages with Sarah Johnson (Feb 12–18):
  Sarah recommended a plumber named Mike Torres on Feb 12, saying he
  fixed her kitchen sink for $150. She texted his number (+1-555-0147)
  on Feb 14 after you asked for it.

Sources: 3 messages, Feb 12–14 2026
```

Everything runs locally. No cloud sync, no telemetry, no accounts.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Discover backups on your system
mudline backups list

# Ingest a backup (extraction + indexing)
mudline backups ingest ~/Library/MobileSync/Backup/<UDID>

# Ask a question
mudline ask "who called me last Tuesday?"

# Structured search
mudline search --type messages --contact Sarah --after 2026-02-01
```

## Architecture

```
Interface (CLI / FastAPI / MCP)
    ↓ query
Intelligence (Query Planner → LLM → Synthesizer)
    ↓ retrieval
Index (ChromaDB vectors + SQLite FTS5 + CLIP media)
    ↓ ingest
Extraction (Domain-specific parsers per iOS data type)
    ↓ read
Foundation (ManifestResolver → raw iOS backup files)
```

**Extraction** parses iOS backup databases into normalized `Document` objects — one extractor per data type (messages, photos, notes, calendar, contacts, calls, voicemail, Safari history).

**Index** stores documents in a hybrid index: ChromaDB for semantic vector search, SQLite FTS5 for exact keyword matching, and CLIP embeddings for image search. A hybrid retriever combines results across all three.

**Intelligence** plans multi-step queries, retrieves relevant documents, and synthesizes cited natural language answers using an LLM. Supports Claude API for quality or Ollama for fully local operation.

**Interface** exposes everything through a CLI (Typer/Rich), a REST API (FastAPI), and an MCP server for integration with AI tools.

## Supported Data Types

| Type | Source | Status |
|------|--------|--------|
| Messages (iMessage/SMS) | `sms.db` | Done |
| Photos & Videos | `Photos.sqlite` + media files | Done |
| Contacts | `AddressBook.sqlitedb` | Done |
| Notes | `NoteStore.sqlite` (protobuf) | Partial — uses snippet, protobuf decode TODO |
| Calendar | `Calendar.sqlitedb` | Done |
| Call History | `CallHistory.storedata` | Done |
| Voicemail | `voicemail.db` + audio files | Done |
| Safari History | `History.db` | Done |

## Development

```bash
# Install with all dependencies
pip install -e ".[dev,media]"

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/

# Test
pytest
```

## Privacy

All data stays on your machine. Mudline never phones home — no analytics, no accounts, no cloud storage. Your backup data is read-only and never modified.

The LLM provider is configurable:
- **Claude API** — best answer quality, requires an API key
- **Ollama** — fully local, no data leaves your machine

## License

MIT
