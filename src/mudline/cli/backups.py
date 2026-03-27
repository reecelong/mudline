"""CLI commands for backup management."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mudline.exceptions import BackupNotFoundError, DecryptionError

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer()


@app.command("list")
def list_backups(
    backup_dir: str | None = typer.Option(
        None,
        "--dir",
        "-d",
        help="Directory to scan for backups (default: standard macOS locations).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Discover and list all iOS backups on this machine."""
    from mudline.cli.main import setup_logging
    from mudline.foundation.discovery import BackupDiscovery

    setup_logging(verbose)

    discovery = BackupDiscovery(verbose=verbose)
    scan_path = Path(backup_dir) if backup_dir else None

    try:
        backups = discovery.discover(scan_path)
    except BackupNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    if not backups:
        console.print("[yellow]No iOS backups found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="iOS Backups")
    table.add_column("Device", style="cyan")
    table.add_column("iOS", style="green")
    table.add_column("UDID", style="dim")
    table.add_column("Date")
    table.add_column("Encrypted")
    table.add_column("Path", style="dim")

    for b in backups:
        table.add_row(
            b.device_name,
            b.ios_version,
            b.udid[:8],
            b.backup_date.strftime("%Y-%m-%d %H:%M"),
            "[red]yes[/red]" if b.is_encrypted else "no",
            str(b.path),
        )

    console.print(table)


@app.command()
def ingest(
    backup_path: str | None = typer.Argument(
        None,
        help="Path to the iOS backup directory. If omitted, auto-discovers the most recent backup.",
    ),
    password: str | None = typer.Option(
        None, "--password", "-p",
        help="Password for encrypted backups (prompted interactively if not provided).",
    ),
    data_dir: str = typer.Option(
        "~/.mudline/data",
        "--data-dir",
        help="Directory for the Mudline index (default: ~/.mudline/data).",
    ),
    no_vectors: bool = typer.Option(
        False, "--no-vectors",
        help="Skip vector embeddings (faster, uses only structured/FTS search).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Ingest an iOS backup into the Mudline index."""
    from mudline.cli.main import setup_logging
    from mudline.extractors.registry import default_registry
    from mudline.foundation.discovery import BackupDiscovery
    from mudline.foundation.manifest import ManifestResolver
    from mudline.index.ingest import IngestPipeline
    from mudline.index.structured import StructuredStore

    setup_logging(verbose)
    data = Path(data_dir).expanduser().resolve()
    data.mkdir(parents=True, exist_ok=True)

    discovery = BackupDiscovery(verbose=verbose)

    if backup_path:
        path = Path(backup_path).expanduser().resolve()
        try:
            info = discovery.validate_backup(path)
        except BackupNotFoundError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from None
    else:
        # Auto-discover the most recent backup
        backups = discovery.discover()
        if not backups:
            console.print(
                "[red]No iOS backups found.[/red] "
                "Provide a path or ensure a backup exists in the default location."
            )
            raise typer.Exit(1)
        info = sorted(backups, key=lambda b: b.backup_date, reverse=True)[0]
        path = info.path
        console.print(f"[dim]Auto-selected most recent backup:[/dim] {path}")

    console.print(
        f"Backup: [cyan]{info.device_name}[/cyan]  "
        f"iOS {info.ios_version}  "
        f"{'[red]encrypted[/red]' if info.is_encrypted else 'unencrypted'}"
    )

    # Set up decryptor if needed
    decryptor = None
    if info.is_encrypted:
        if not password:
            password = typer.prompt("Backup password", hide_input=True)
        try:
            from mudline.foundation.crypto import KeybagDecryptor

            decryptor = KeybagDecryptor(path, password)
        except DecryptionError as exc:
            console.print(f"[red]Decryption failed:[/red] {exc}")
            raise typer.Exit(1) from None

    try:
        resolver = ManifestResolver(path, decryptor=decryptor)

        # Storage
        structured_store = StructuredStore(data / "index.db")

        vector_store = None
        if no_vectors:
            console.print("[dim]Skipping vector embeddings (--no-vectors)[/dim]")
        else:
            try:
                from mudline.index.vector import VectorStore, VectorStoreConfig

                vector_store = VectorStore(
                    VectorStoreConfig(persist_directory=data / "vectors")
                )
            except Exception as exc:
                console.print(
                    f"[yellow]Warning:[/yellow] Vector store unavailable ({exc}); "
                    "only structured search will work."
                )

        pipeline = IngestPipeline(structured_store, vector_store)
        base_backup_id = f"{info.udid}:{info.backup_date.isoformat()}"

        # Run extractors — each gets its own backup_id for independent state tracking
        extractors = default_registry.create_all()
        summary: dict[str, int] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed} docs"),
            console=console,
        ) as progress:
            for extractor in extractors:
                if not extractor.can_extract(resolver):
                    logger.debug("Skipping %s (data not present)", extractor.data_type)
                    continue

                task = progress.add_task(
                    f"[cyan]{extractor.data_type}[/cyan]", total=None
                )
                count = 0

                def _on_progress(total: int, _task: int = task) -> None:
                    nonlocal count
                    progress.update(_task, completed=total)
                    count = total

                extractor_backup_id = f"{base_backup_id}:{extractor.data_type}"
                try:
                    state = pipeline.ingest(
                        extractor_backup_id,
                        extractor.extract(resolver),
                        on_progress=_on_progress,
                    )
                    count = state.document_count
                except Exception as exc:
                    console.print(
                        f"[red]Error extracting {extractor.data_type}:[/red] {exc}"
                    )

                summary[extractor.data_type] = count
                progress.update(task, completed=count)

        # Print summary
        console.print()
        total = sum(summary.values())
        summary_table = Table(title="Ingest Summary")
        summary_table.add_column("Type", style="cyan")
        summary_table.add_column("Documents", justify="right")
        for dtype, cnt in sorted(summary.items()):
            summary_table.add_row(dtype, str(cnt))
        summary_table.add_section()
        summary_table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
        console.print(summary_table)

    finally:
        if decryptor is not None:
            decryptor.close()


@app.command()
def status(
    data_dir: str = typer.Option(
        "~/.mudline/data",
        "--data-dir",
        help="Directory for the Mudline index (default: ~/.mudline/data).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Show the current state of the Mudline index."""
    from mudline.cli.main import setup_logging
    from mudline.index.structured import StructuredStore

    setup_logging(verbose)
    data = Path(data_dir).expanduser().resolve()
    db_path = data / "index.db"

    if not db_path.exists():
        console.print("[yellow]No index found.[/yellow] Run [bold]mudline backups ingest[/bold] first.")
        raise typer.Exit(0)

    store = StructuredStore(db_path)

    # Document counts by type
    import sqlite3

    conn = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        cursor = conn.execute(
            "SELECT type, COUNT(*) FROM documents GROUP BY type ORDER BY type"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        console.print("[yellow]Index exists but contains no documents.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Index Status")
    table.add_column("Type", style="cyan")
    table.add_column("Documents", justify="right")

    total = 0
    for doc_type, count in rows:
        table.add_row(doc_type, str(count))
        total += count

    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
    console.print(table)

    # Ingest state
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        cursor = conn.execute(
            "SELECT backup_id, status, document_count, started_at, completed_at "
            "FROM _ingest_state ORDER BY started_at DESC"
        )
        ingest_rows = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError:
        ingest_rows = []

    if ingest_rows:
        console.print()
        ingest_table = Table(title="Ingest History")
        ingest_table.add_column("Backup ID", style="dim")
        ingest_table.add_column("Status")
        ingest_table.add_column("Documents", justify="right")
        ingest_table.add_column("Started")
        ingest_table.add_column("Completed")

        for row in ingest_rows:
            bid, st, cnt, started, completed = row
            status_style = {"completed": "green", "failed": "red", "in_progress": "yellow"}.get(
                st, ""
            )
            ingest_table.add_row(
                bid[:20] + "..." if len(bid) > 20 else bid,
                f"[{status_style}]{st}[/{status_style}]",
                str(cnt),
                started or "",
                completed or "",
            )

        console.print(ingest_table)
