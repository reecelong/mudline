# Mudline — Task Breakdown

## Execution Waves

**Wave 0 (no dependencies, start immediately):**
C-01, F-01, I-01, I-02, Q-06

**Wave 1 (after F-02):**
E-01, E-02, E-03, E-04, E-05, E-06, E-07, E-08

**Wave 2 (after extraction + index stores):**
F-03, F-04, I-03, I-04, I-05, I-06, E-09

**Wave 3 (after index layer):**
Q-01, Q-02, Q-03, Q-04, Q-05, C-02, C-03

**Wave 4 (final assembly):**
C-04, C-05, C-06

---

## Foundation Agent

### F-01: Backup Discovery
- **Deps:** None
- **Effort:** S (1-2 hrs)
- **Description:** Scan filesystem for valid iOS backup directories. Parse `Info.plist` and `Status.plist` to identify device name, iOS version, UDID, backup date, encrypted status.
- **Output:** `mudline.foundation.discovery.BackupDiscovery` class, `BackupInfo` dataclass
- **Acceptance:**
  - Discovers backups in `~/Library/MobileSync/Backup/` (macOS) and custom paths
  - Returns `BackupInfo` with: device_name, ios_version, udid, backup_date, is_encrypted, path
  - Handles missing/corrupt plists gracefully with logging
  - Unit tests with a minimal synthetic backup fixture

### F-02: ManifestResolver
- **Deps:** F-01
- **Effort:** M (2-4 hrs)
- **Description:** Parse `Manifest.db` to map SHA-1 file hashes to domain + relative path. Implement `resolve(domain, path) -> Path` and `list_domain(domain) -> list`.
- **Output:** `mudline.foundation.manifest.ManifestResolver`
- **Acceptance:**
  - `resolve("HomeDomain", "Library/SMS/sms.db")` returns the correct file path
  - `list_domain("HomeDomain")` returns all files in that domain
  - `list_domains()` returns all unique domains in the backup
  - Handles missing files with `FileNotFoundError`
  - Unit tests against synthetic Manifest.db fixture

### F-03: Encrypted Backup Support
- **Deps:** F-02
- **Effort:** L (4-8 hrs)
- **Description:** Implement keybag parsing from `Manifest.plist`. Derive encryption keys from user password. Decrypt individual files on demand via the resolver.
- **Output:** `mudline.foundation.crypto.KeybagDecryptor`, integration with ManifestResolver
- **Acceptance:**
  - Given a password, decrypts files from an encrypted backup
  - Incorrect password raises `DecryptionError`
  - Decrypted files match known plaintext from test fixture
  - Uses `iOSbackup` library internals as reference

### F-04: Multi-Backup Manager
- **Deps:** F-02
- **Effort:** M (2-4 hrs)
- **Description:** Track multiple backup snapshots for the same device. Provide API to enumerate snapshots and compare file lists between them.
- **Output:** `mudline.foundation.manager.BackupManager`
- **Acceptance:**
  - Groups backups by UDID
  - Lists snapshots in chronological order
  - `diff(snapshot_a, snapshot_b)` returns added/removed/changed files

---

## Extraction Agent

All extractors depend on F-02 (ManifestResolver) and produce `Iterator[Document]`.

### E-01: MessageExtractor (iMessage/SMS)
- **Deps:** F-02
- **Effort:** L (4-8 hrs)
- **Description:** Parse `HomeDomain/Library/SMS/sms.db`. Extract messages with: text, timestamp (Cocoa epoch → datetime), is_from_me, handle (phone/email), chat display_name, thread grouping. Handle attachment references.
- **Output:** `mudline.extractors.messages.MessageExtractor`
- **Acceptance:**
  - Extracts all messages with correct timestamp conversion
  - Groups messages by conversation thread (chat_id)
  - Resolves handles to phone numbers / email addresses
  - Attachment metadata included in Document.attachments
  - Handles group chats (multiple participants)

### E-02: ContactExtractor
- **Deps:** F-02
- **Effort:** M (2-4 hrs)
- **Description:** Parse `HomeDomain/Library/AddressBook/AddressBook.sqlitedb`. Extract contacts with all phone numbers, emails, addresses, organization. Build handle→contact resolution map.
- **Output:** `mudline.extractors.contacts.ContactExtractor`
- **Acceptance:**
  - Extracts all contacts with multi-value fields (multiple phones, emails)
  - Produces a `handle_map: dict[str, str]` mapping phone/email → contact display name
  - Handles contacts with no name (company-only entries)

### E-03: PhotoExtractor
- **Deps:** F-02
- **Effort:** M (2-4 hrs)
- **Description:** Parse `CameraRollDomain/Media/PhotoData/Photos.sqlite`. Extract metadata: filename, creation date, GPS coordinates, dimensions, albums. Resolve actual image file paths through manifest.
- **Output:** `mudline.extractors.photos.PhotoExtractor`
- **Acceptance:**
  - Extracts photo metadata with GPS when available
  - Resolves image file paths for downstream CLIP indexing
  - Handles HEIC, JPEG, PNG, and video thumbnails
  - Albums included in metadata

### E-04: NoteExtractor
- **Deps:** F-02
- **Effort:** L (4-8 hrs)
- **Description:** Parse `HomeDomain/Library/Notes/NoteStore.sqlite`. Decode protobuf-encoded rich text from `ZICCLOUDSYNCINGOBJECT` into plain text. Extract title, body, dates, folder hierarchy.
- **Output:** `mudline.extractors.notes.NoteExtractor`
- **Acceptance:**
  - Decodes protobuf note content into readable plain text
  - Extracts title (first line or explicit title field)
  - Preserves folder/subfolder hierarchy in metadata
  - Handles notes with embedded images (references in attachments)

### E-05: CalendarExtractor
- **Deps:** F-02
- **Effort:** M (2-4 hrs)
- **Description:** Parse `HomeDomain/Library/Calendar/Calendar.sqlitedb`. Extract events: title, start/end time, location, notes, attendees, recurrence.
- **Output:** `mudline.extractors.calendar.CalendarExtractor`
- **Acceptance:**
  - Extracts events with correct timezone handling
  - Includes location and attendee data in metadata
  - Handles all-day events and recurring events

### E-06: CallHistoryExtractor
- **Deps:** F-02
- **Effort:** S (1-2 hrs)
- **Description:** Parse `HomeDomain/Library/CallHistoryDB/CallHistory.storedata`. Extract: contact, duration, timestamp, call type.
- **Output:** `mudline.extractors.calls.CallHistoryExtractor`
- **Acceptance:**
  - Extracts calls with type (incoming/outgoing/missed)
  - Duration in seconds
  - Handle for contact cross-referencing

### E-07: SafariExtractor
- **Deps:** F-02
- **Effort:** S (1-2 hrs)
- **Description:** Parse Safari `Bookmarks.db` and `History.db`. Extract browsing history and bookmarks.
- **Output:** `mudline.extractors.safari.SafariExtractor`
- **Acceptance:**
  - History: URL, title, visit timestamp, visit count
  - Bookmarks: URL, title, folder structure

### E-08: VoicemailExtractor
- **Deps:** F-02
- **Effort:** S (1-2 hrs)
- **Description:** Parse `HomeDomain/Library/Voicemail/voicemail.db`. Extract metadata and transcription text. Resolve audio file references.
- **Output:** `mudline.extractors.voicemail.VoicemailExtractor`
- **Acceptance:**
  - Extracts sender, date, duration, transcription
  - Audio file path in attachments

### E-09: Extractor Registry
- **Deps:** E-01 (needs at least one extractor to test)
- **Effort:** S (1-2 hrs)
- **Description:** Auto-discovery of extractors via a registry pattern. CLI-listable.
- **Output:** `mudline.extractors.registry.ExtractorRegistry`
- **Acceptance:**
  - `registry.list()` returns all available extractors
  - `registry.get("messages")` returns MessageExtractor
  - New extractors auto-register by inheriting from Extractor protocol

---

## Index Agent

### I-01: Structured Store
- **Deps:** None
- **Effort:** M (2-4 hrs)
- **Description:** SQLite schema with FTS5 for full-text search. Normalized documents table with type, timestamp, contact_handle, thread_id, domain columns. JSONB metadata.
- **Output:** `mudline.index.structured.StructuredStore`
- **Acceptance:**
  - `insert(doc: Document)` and `insert_batch(docs: list[Document])`
  - `query(type=, contact=, after=, before=, text_search=)` with any combination of filters
  - FTS5 search returns ranked results
  - Handles deduplication by document ID

### I-02: Vector Store Setup
- **Deps:** None
- **Effort:** S (1-2 hrs)
- **Description:** ChromaDB with persistent storage. Embedding model configuration. Collection schema with metadata fields.
- **Output:** `mudline.index.vector.VectorStore`
- **Acceptance:**
  - `add(docs: list[Document])` embeds and stores
  - `query(text: str, where: dict, n: int)` returns scored results
  - Persistence across process restarts
  - Configurable embedding model

### I-03: Ingest Pipeline
- **Deps:** I-01, I-02
- **Effort:** L (4-8 hrs)
- **Description:** Consumes `Iterator[Document]` from extractors. Batch-embeds and writes to both stores atomically. Tracks ingestion state for incremental re-ingestion.
- **Output:** `mudline.index.ingest.IngestPipeline`
- **Acceptance:**
  - `ingest(backup_id: str, documents: Iterator[Document])` populates both stores
  - Tracks which backup_ids have been ingested
  - Re-ingestion of same backup is idempotent
  - Progress reporting via callback

### I-04: Hybrid Retriever
- **Deps:** I-01, I-02
- **Effort:** L (4-8 hrs)
- **Description:** Combines structured SQL filtering with vector similarity ranking. Supports pure-structured, pure-semantic, and hybrid modes.
- **Output:** `mudline.index.retriever.HybridRetriever` implementing `Retriever` protocol
- **Acceptance:**
  - `search(query, filters, limit)` returns scored Results with source provenance
  - Structured-only: correct SQL filtering without vector search
  - Semantic-only: correct vector ranking without SQL filters
  - Hybrid: SQL narrows, vector ranks within filtered set
  - Empty filter sets handled correctly

### I-05: Media Index (CLIP)
- **Deps:** I-02, E-03
- **Effort:** L (4-8 hrs)
- **Description:** CLIP embeddings for photos. Separate ChromaDB collection. Visual semantic search.
- **Output:** `mudline.index.media.MediaIndex`
- **Acceptance:**
  - `index_photo(path: Path, metadata: dict)` generates CLIP embedding
  - `search(query: str, n: int)` returns photos ranked by visual similarity
  - Handles HEIC conversion for CLIP input

### I-06: Contact Resolution Index
- **Deps:** E-02
- **Effort:** M (2-4 hrs)
- **Description:** Unified contact graph mapping phone numbers, emails, iMessage handles to resolved names.
- **Output:** `mudline.index.contacts.ContactIndex`
- **Acceptance:**
  - `resolve(name: str) -> list[str]` returns all handles for "John"
  - `lookup(handle: str) -> str | None` returns display name for a handle
  - Fuzzy name matching (John, john, Johnny → same contact)

---

## Intelligence Agent

### Q-01: Tool Definitions
- **Deps:** I-04
- **Effort:** M (2-4 hrs)
- **Description:** Function-calling tool schemas: search_messages, search_photos, search_notes, get_contact, get_call_history, search_calendar, search_safari.
- **Output:** `mudline.intelligence.tools` module
- **Acceptance:**
  - Each tool has a JSON Schema definition compatible with Anthropic/OpenAI function calling
  - Each tool maps to a Retriever call with appropriate filters
  - Tool execution returns structured results

### Q-02: Query Planner
- **Deps:** Q-01, I-06
- **Effort:** L (4-8 hrs)
- **Description:** LLM-powered decomposition of natural language into tool calls. Temporal reasoning ("last week" → date range). Ambiguity handling ("John" → contact resolution).
- **Output:** `mudline.intelligence.planner.QueryPlanner`
- **Acceptance:**
  - "What did John text me about the plumber?" → resolve_contact + search_messages
  - "Photos from the beach last summer" → search_photos with date range + query
  - "Show me my calendar for next week" → search_calendar with date range
  - Handles ambiguous contact names by asking for clarification
  - Temporal expressions resolved correctly relative to current date

### Q-03: Context Expansion
- **Deps:** I-04
- **Effort:** M (2-4 hrs)
- **Description:** Expand retrieved messages to surrounding conversation context. Include full note bodies. Timeline-adjacent photos.
- **Output:** `mudline.intelligence.context.ContextExpander`
- **Acceptance:**
  - Given a message result, returns N messages before/after in same thread
  - Given a note result, returns full note body
  - Given a photo result, returns temporally adjacent photos

### Q-04: Synthesizer
- **Deps:** Q-03
- **Effort:** M (2-4 hrs)
- **Description:** LLM that assembles retrieved + expanded results into natural language answers with citations.
- **Output:** `mudline.intelligence.synthesizer.Synthesizer`
- **Acceptance:**
  - Produces conversational answers referencing specific messages/dates/contacts
  - Includes source citations: backup timestamp, conversation, message date
  - Handles multi-part results ("Here are 3 relevant conversations...")

### Q-05: Conversation Memory
- **Deps:** Q-04
- **Effort:** M (2-4 hrs)
- **Description:** Multi-turn conversation state. Follow-up queries retain context from previous turns.
- **Output:** `mudline.intelligence.memory.ConversationMemory`
- **Acceptance:**
  - "What about last Tuesday?" after asking about John retains the John filter
  - Sliding window context management (last N turns)
  - Memory can be cleared explicitly

### Q-06: LLM Provider Abstraction
- **Deps:** None
- **Effort:** M (2-4 hrs)
- **Description:** Support multiple LLM backends: Anthropic Claude, OpenAI-compatible (Ollama), configurable.
- **Output:** `mudline.intelligence.llm.LLMProvider`, `mudline.intelligence.llm.AnthropicProvider`, `mudline.intelligence.llm.OllamaProvider`
- **Acceptance:**
  - Unified interface: `complete(messages, tools) -> Response`
  - Anthropic provider uses `anthropic` SDK with function calling
  - Ollama provider uses OpenAI-compatible API at configurable URL
  - Provider selected via config file or environment variable

---

## Interface Agent

### C-01: Project Scaffolding
- **Deps:** None
- **Effort:** S (1-2 hrs)
- **Description:** Python project setup: pyproject.toml, src layout, dev tooling.
- **Output:** Complete project skeleton with CI config
- **Acceptance:**
  - `pip install -e .` works
  - `ruff check .` passes
  - `mypy src/` passes
  - `pytest` discovers and runs tests

### C-02: CLI Backup Management
- **Deps:** F-01, I-03
- **Effort:** M (2-4 hrs)
- **Description:** Commands: `mudline backups list`, `mudline backups ingest <path>`, `mudline backups status`.
- **Output:** `mudline.cli.backups` module
- **Acceptance:**
  - `mudline backups list` shows discovered backups with device info
  - `mudline backups ingest ~/path` triggers full extraction + indexing pipeline
  - `mudline backups status` shows ingest progress and stats
  - Rich-formatted output

### C-03: CLI Search
- **Deps:** I-04, Q-04
- **Effort:** M (2-4 hrs)
- **Description:** `mudline search <query>` (structured) and `mudline ask <question>` (natural language).
- **Output:** `mudline.cli.search` module
- **Acceptance:**
  - `mudline search --type messages --contact John` returns filtered results
  - `mudline ask "what did John say about the plumber?"` returns NL answer
  - Results formatted with timestamps, contacts, excerpts

### C-04: Interactive TUI
- **Deps:** C-03, Q-05
- **Effort:** L (4-8 hrs)
- **Description:** Textual-based interactive session with streaming responses and follow-ups.
- **Output:** `mudline.cli.tui` module
- **Acceptance:**
  - Persistent conversation with follow-up context
  - Streaming LLM response display
  - Sidebar with backup/device info
  - `/clear` to reset conversation

### C-05: FastAPI Server
- **Deps:** Q-04
- **Effort:** L (4-8 hrs)
- **Description:** REST API + WebSocket for streaming responses.
- **Output:** `mudline.api.server` module
- **Acceptance:**
  - `POST /ask` with NL query, returns streaming JSON
  - `GET /search` with structured filters
  - `GET /backups` lists ingested backups
  - `GET /documents/{id}` returns full document
  - WebSocket endpoint for interactive sessions

### C-06: MCP Server Wrapper
- **Deps:** C-05
- **Effort:** M (2-4 hrs)
- **Description:** MCP tool server wrapping Recall's capabilities for Claude Code / agent consumption.
- **Output:** `mudline.api.mcp` module
- **Acceptance:**
  - Tools: `mudline_ask`, `mudline_search`, `mudline_list_contacts`, `mudline_get_conversation`
  - Works with `claude mcp add` for Claude Code integration
  - Returns structured JSON suitable for LLM consumption
