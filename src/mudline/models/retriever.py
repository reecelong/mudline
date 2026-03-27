"""Retriever interface — the unified search API consumed by the intelligence layer.

The query planner calls retrieval through this interface. The hybrid retriever
implements it by combining SQL filtering with vector similarity ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime

    from mudline.models.document import Document, DocumentType


@dataclass
class Filters:
    """Structured filters for narrowing search results.

    All fields are optional. When multiple fields are set, they are ANDed together.
    """
    data_types: list[DocumentType] | None = None  # Filter to specific types
    contacts: list[str] | None = None              # Filter by contact handles
    date_after: datetime | None = None             # Documents after this date
    date_before: datetime | None = None            # Documents before this date
    has_attachments: bool | None = None            # Only docs with/without attachments
    backup_id: str | None = None                   # Restrict to specific backup
    thread_id: int | None = None                   # Specific conversation thread
    metadata: dict[str, str] | None = None         # Arbitrary metadata key-value filters


@dataclass
class Result:
    """A single search result with scoring and provenance."""
    document: Document
    score: float                                   # 0.0-1.0, higher is more relevant
    highlights: list[str] = field(default_factory=list)  # Matching text snippets
    match_type: str = "hybrid"                     # "structured", "semantic", or "hybrid"


@runtime_checkable
class Retriever(Protocol):
    """Protocol for the search/retrieval layer.

    Implementations must support pure structured queries, pure semantic queries,
    and hybrid queries that combine both.
    """

    def search(
        self,
        query: str | None = None,
        filters: Filters | None = None,
        limit: int = 10,
    ) -> list[Result]:
        """Search for documents matching the query and/or filters.

        Args:
            query: Natural language query for semantic search. None for structured-only.
            filters: Structured filters. None for semantic-only.
            limit: Maximum number of results to return.

        Returns:
            List of Results sorted by descending score.

        Raises:
            SearchError: If the search infrastructure is unavailable.
        """
        ...

    def get_by_id(self, document_id: str) -> Document | None:
        """Retrieve a specific document by its ID.

        Returns:
            The Document, or None if not found.
        """
        ...

    def get_thread(
        self,
        thread_id: int,
        around_timestamp: datetime | None = None,
        window: int = 10,
    ) -> list[Document]:
        """Retrieve messages in a conversation thread.

        Args:
            thread_id: The conversation thread ID.
            around_timestamp: Center the window around this time. None for latest.
            window: Number of messages to return (split evenly before/after).

        Returns:
            List of Documents in chronological order.
        """
        ...
