"""Mudline CLI — AI-powered personal data layer over iOS backups."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

# Load .env from ~/.mudline/.env, then project root, then cwd
load_dotenv(Path.home() / ".mudline" / ".env")
load_dotenv()

app = typer.Typer(
    name="mudline",
    help="AI-powered personal data layer over iOS backups.",
    no_args_is_help=True,
)


def setup_logging(verbose: bool) -> None:
    """Configure root logging level based on the verbose flag."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s: %(message)s")


# Register subcommands
from mudline.cli import backups as _backups_mod  # noqa: E402
from mudline.cli import search as _search_mod  # noqa: E402

app.add_typer(_backups_mod.app, name="backups", help="Manage iOS backups.")
app.add_typer(_search_mod.app, name="query", help="Search and ask questions.")

# Top-level shortcuts so users can run `mudline search` / `mudline ask` directly
app.command(name="search")(_search_mod.search)
app.command(name="ask")(_search_mod.ask)

if __name__ == "__main__":
    app()
