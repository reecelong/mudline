"""C-03: CLI Search — structured search and natural language question answering.

Provides two commands:
- `mudline search`: Structured search with explicit type, contact, and date filters.
- `mudline ask`: Natural language question answering via the intelligence layer.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from mudline.models.document import DocumentType

app = typer.Typer()
console = Console()

DEFAULT_DATA_DIR = Path.home() / ".mudline" / "data"

logger = logging.getLogger(__name__)


def _open_stores(data_dir: Path) -> tuple:
    """Open index stores, returning (StructuredStore, VectorStore | None).

    Args:
        data_dir: Directory containing the index database and vector data.

    Returns:
        Tuple of (structured_store, vector_store_or_none).

    Raises:
        typer.Exit: If no index database exists at the expected path.
    """
    from mudline.index.structured import StructuredStore

    db_path = data_dir / "index.db"
    if not db_path.exists():
        console.print("[red]No index found. Run 'mudline backups ingest' first.[/red]")
        raise typer.Exit(1)

    structured = StructuredStore(db_path)

    vector = None
    try:
        from mudline.index.vector import VectorStore, VectorStoreConfig

        vector_dir = data_dir / "vectors"
        if vector_dir.exists():
            vector = VectorStore(VectorStoreConfig(persist_directory=vector_dir))
    except Exception:
        pass

    return structured, vector


def _parse_date(value: str) -> datetime:
    """Parse a date string in ISO format.

    Args:
        value: Date string like "2025-01-15" or "2025-01-15T10:30:00".

    Returns:
        Parsed datetime object.

    Raises:
        typer.BadParameter: If the date string cannot be parsed.
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError as e:
        raise typer.BadParameter(
            f"Invalid date format: {value!r}. Use ISO format (YYYY-MM-DD)."
        ) from e


def _get_backup_date(data_dir: Path) -> str | None:
    """Get the backup date from the most recently ingested backup's Info.plist."""
    import sqlite3

    try:
        conn = sqlite3.connect(str(data_dir / "index.db"), timeout=5.0)
        row = conn.execute(
            "SELECT backup_id FROM _ingest_state ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if not row:
            return None
        # backup_id format: "UDID:ISO_TIMESTAMP:extractor_type"
        # The timestamp is the backup date from Info.plist
        parts = row[0].split(":")
        if len(parts) >= 2:
            # Rejoin the ISO timestamp parts (date:time)
            iso_ts = ":".join(parts[1:-1]) if len(parts) > 2 else parts[1]
            # Convert UTC to local time
            from datetime import timezone

            utc_dt = datetime.fromisoformat(iso_ts).replace(tzinfo=timezone.utc)
            local_dt = utc_dt.astimezone()
            return local_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    return None


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text to max_len, appending ellipsis if shortened."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


@app.command()
def search(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Text search query."),
    type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Document type filter (message, note, photo, contact, calendar, call, voicemail, safari).",
    ),
    contact: Optional[str] = typer.Option(None, "--contact", "-c", help="Contact name or handle."),
    after: Optional[str] = typer.Option(
        None, "--after", "-a", help="Show results after this date (ISO format)."
    ),
    before: Optional[str] = typer.Option(
        None, "--before", "-b", help="Show results before this date (ISO format)."
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results."),
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, "--data-dir", help="Directory containing index data."
    ),
) -> None:
    """Search indexed iOS backup data with structured filters."""
    structured, vector = _open_stores(data_dir)

    type_filter: DocumentType | None = None
    if type:
        try:
            type_filter = DocumentType(type.lower())
        except ValueError:
            valid = ", ".join(t.value for t in DocumentType)
            console.print(f"[red]Invalid type: {type!r}. Valid types: {valid}[/red]")
            raise typer.Exit(1)

    after_dt = _parse_date(after) if after else None
    before_dt = _parse_date(before) if before else None

    if not any([query, type_filter, contact, after_dt, before_dt]):
        console.print(
            "[yellow]Provide at least one filter "
            "(--query, --type, --contact, --after, --before).[/yellow]"
        )
        raise typer.Exit(1)

    from mudline.index.retriever import HybridRetriever
    from mudline.models.retriever import Filters

    retriever = HybridRetriever(structured, vector)

    filters = Filters(
        data_types=[type_filter] if type_filter else None,
        contacts=[contact] if contact else None,
        date_after=after_dt,
        date_before=before_dt,
    )

    results = retriever.search(query=query, filters=filters, limit=limit)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        latest = _get_latest_backup_timestamp(structured)
        if latest:
            console.print(f"[dim]Backup date: {latest}[/dim]")
        raise typer.Exit(0)

    table = Table(title=f"Search Results ({len(results)} found)")
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Timestamp", style="green", no_wrap=True)
    table.add_column("Contact/Handle", style="magenta")
    table.add_column("Text", style="white")
    table.add_column("Score", style="yellow", justify="right")

    for result in results:
        doc = result.document
        ts = doc.timestamp.strftime("%Y-%m-%d %H:%M") if doc.timestamp else ""
        handle = doc.metadata.get("handle", "")
        table.add_row(
            doc.type.value,
            ts,
            handle,
            _truncate(doc.text),
            f"{result.score:.2f}",
        )

    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural language question about your data."),
    provider: str = typer.Option(
        "claude-code",
        "--provider",
        help="LLM provider: claude-code (default, uses CLI), anthropic, or ollama.",
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Model name override."),
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, "--data-dir", help="Directory containing index data."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show debug info including citations and token usage."
    ),
) -> None:
    """Ask a natural language question about your iOS backup data."""
    asyncio.run(_ask_async(question, provider, model, data_dir, verbose))


async def _ask_async(
    question: str,
    provider: str,
    model: str | None,
    data_dir: Path,
    verbose: bool,
) -> None:
    """Async implementation of the ask command.

    Routes queries through a fast path (structured search, no LLM) when
    possible, and only escalates to the full planner pipeline when the
    query requires reasoning or synthesis.

    Args:
        question: The user's natural language question.
        provider: LLM provider name ("claude-code", "anthropic", or "ollama").
        model: Optional model name override.
        data_dir: Directory containing the index data.
        verbose: Whether to show citations and token usage.
    """
    structured, vector = _open_stores(data_dir)

    from mudline.index.contacts import ContactIndex
    from mudline.index.retriever import HybridRetriever
    from mudline.intelligence.router import QueryRouter

    retriever = HybridRetriever(structured, vector)

    contact_index = ContactIndex()
    contact_docs = structured.query(type_filter=DocumentType.CONTACT, limit=10000)
    contact_index.load_from_documents(contact_docs)

    # Route: fast path or LLM?
    router = QueryRouter(contact_index)
    route = router.route(question)

    if verbose:
        console.print(f"[dim]Route: {route.reason}[/dim]")

    if not route.needs_llm:
        # Fast path — structured search, no LLM calls
        results = retriever.search(query=route.search_query, filters=route.filters, limit=20)

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            latest = _get_backup_date(data_dir)
            if latest:
                console.print(f"[dim]Backup date: {latest}[/dim]")
            raise typer.Exit(0)

        table = Table(title=f"Results ({len(results)} found)")
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Timestamp", style="green", no_wrap=True)
        table.add_column("Contact", style="magenta")
        table.add_column("Text", style="white")
        table.add_column("Score", style="yellow", justify="right")

        for result in results:
            doc = result.document
            ts = doc.timestamp.strftime("%Y-%m-%d %H:%M") if doc.timestamp else ""
            handle = doc.metadata.get("handle", "")
            # Resolve handle to name if possible
            name = contact_index.lookup(handle) if handle else None
            display_contact = name or handle
            table.add_row(
                doc.type.value,
                ts,
                display_contact,
                _truncate(doc.text),
                f"{result.score:.2f}",
            )

        console.print(table)
        return

    # Slow path — full LLM pipeline
    from mudline.intelligence.context import ContextExpander
    from mudline.intelligence.llm import create_provider
    from mudline.intelligence.planner import QueryPlanner
    from mudline.intelligence.synthesizer import Synthesizer
    from mudline.intelligence.tools import ToolRegistry

    tool_registry = ToolRegistry(retriever, contact_index)

    llm_kwargs: dict[str, str] = {}
    if model:
        llm_kwargs["model"] = model

    try:
        llm = create_provider(provider, **llm_kwargs)
    except Exception as e:
        console.print(f"[red]Failed to initialize LLM provider ({provider}): {e}[/red]")
        raise typer.Exit(1)

    planner = QueryPlanner(llm, tool_registry)
    context_expander = ContextExpander(retriever)
    synthesizer = Synthesizer(llm)

    with console.status("[bold cyan]Thinking...[/bold cyan]"):
        try:
            plan_result = await planner.plan_and_execute(question)
        except Exception as e:
            console.print(f"[red]Query planning failed: {e}[/red]")
            raise typer.Exit(1)

        expanded = None
        all_docs: list[dict] = []
        for results_list in plan_result.tool_results.values():
            for item in results_list:
                if "error" not in item and "id" in item:
                    all_docs.append(item)

        if all_docs:
            from mudline.models.retriever import Result

            results_for_expansion: list[Result] = []
            for item in all_docs:
                doc = structured.get_by_id(item["id"])
                if doc:
                    results_for_expansion.append(Result(document=doc, score=item.get("score", 1.0)))

            if results_for_expansion:
                expanded = context_expander.expand_batch(results_for_expansion)

        try:
            answer = await synthesizer.synthesize(question, plan_result, expanded)
        except Exception as e:
            console.print(f"[red]Synthesis failed: {e}[/red]")
            raise typer.Exit(1)

    console.print()
    console.print(Markdown(answer.text))
    console.print()

    if verbose:
        if answer.citations:
            citation_table = Table(title="Citations")
            citation_table.add_column("Type", style="cyan")
            citation_table.add_column("Timestamp", style="green")
            citation_table.add_column("Contact", style="magenta")
            citation_table.add_column("Excerpt", style="white")

            for citation in answer.citations:
                citation_table.add_row(
                    citation.document_type,
                    citation.timestamp or "",
                    citation.contact or "",
                    _truncate(citation.excerpt, 60),
                )

            console.print(citation_table)
            console.print()

        if answer.usage:
            input_tokens = answer.usage.get("input_tokens", 0)
            output_tokens = answer.usage.get("output_tokens", 0)
            console.print(
                f"[dim]Tokens: {input_tokens} input, {output_tokens} output "
                f"({input_tokens + output_tokens} total)[/dim]"
            )
