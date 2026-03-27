# Mudline Project Plugin

> Manage and execute the Mudline implementation plan. Use this when working on any Mudline task.

## Instructions

You are working on **Mudline**, an AI-powered personal data layer over iOS backups. The project is decomposed into 27 tasks across 5 agent roles (Foundation, Extraction, Index, Intelligence, Interface).

### Before starting any task:

1. Read `TASKS.md` to find the task description and acceptance criteria
2. Check the task's dependencies — do not start if deps aren't merged
3. Read the shared contracts in `src/mudline/models/` — your code must conform to these interfaces
4. Read `CLAUDE.md` for code conventions

### When implementing a task:

1. Create the implementation file in the correct package (e.g., `src/mudline/extractors/messages.py` for E-01)
2. Write tests in the corresponding test directory (e.g., `tests/extractors/test_messages.py`)
3. Run `ruff check src/ tests/` before considering the task complete
4. Verify all tests pass with `pytest`
5. Mark the task as done by appending `✅` to its heading in TASKS.md

### Task execution order (respect dependency waves):

- **Wave 0** (start now): C-01, F-01, I-01, I-02, Q-06
- **Wave 1** (after F-02): E-01 through E-08
- **Wave 2** (after extraction + stores): F-03, F-04, I-03, I-04, I-05, I-06, E-09
- **Wave 3** (after index): Q-01 through Q-05, C-02, C-03
- **Wave 4** (final): C-04, C-05, C-06

### Key technical reminders:

- iOS timestamps use Cocoa epoch (seconds since 2001-01-01)
- Backup files stored in 2-char SHA-1 prefix subdirectories
- Notes use protobuf encoding, not plain text
- Every extractor must implement the `Extractor` protocol from `src/mudline/models/extractor.py`
- Every extractor must yield `Document` objects from `src/mudline/models/document.py`

## Slash Commands

- `/mudline-status` — Show which tasks are complete, in progress, and blocked
- `/mudline-next` — Suggest the highest-priority unblocked task to work on
- `/mudline-test` — Run the full test suite and report results
