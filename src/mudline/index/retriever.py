"""Hybrid retriever combining structured and semantic search.

This module implements the Retriever protocol by combining SQL-based structured
queries with optional vector-based semantic search for ranking.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from mudline.exceptions import SearchError
from mudline.models.retriever import Filters, Result

if TYPE_CHECKING:
    from mudline.index.structured import StructuredStore
    from mudline.index.vector import VectorStore
    from mudline.models.document import Document

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining structured and semantic search.

    Implements the Retriever protocol by supporting:
    - Pure structured queries (filters only) using SQL
    - Pure semantic queries (text only) using vector similarity
    - Hybrid queries (both text and filters) using combined scoring

    Args:
        structured_store: SQLite-backed structured store for filtered queries.
        vector_store: Optional VectorStore for semantic ranking. If None, falls
                      back to full-text search (FTS) in structured store.
    """

    def __init__(
        self,
        structured_store: StructuredStore,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            structured_store: Required SQLite store.
            vector_store: Optional vector store for embeddings.
        """
        self.structured_store = structured_store
        self.vector_store = vector_store

    def search(
        self,
        query: str | None = None,
        filters: Filters | None = None,
        limit: int = 10,
    ) -> list[Result]:
        """Search for documents matching query and/or filters.

        Strategy:
        - If only filters (no query) → structured query with score=1.0
        - If only query (no filters) → semantic search or FTS fallback
        - If both → structured query then re-rank with semantic scores

        Args:
            query: Natural language query for semantic search. None for structured-only.
            filters: Structured filters. None for semantic-only.
            limit: Maximum number of results to return.

        Returns:
            List of Result objects sorted by descending score.

        Raises:
            SearchError: If search fails.
        """
        try:
            # Case 1: Structured-only search (filters, no query text)
            if not query and filters:
                return self._search_structured_only(filters, limit)

            # Case 2: Semantic-only search (query text, no filters)
            if query and not filters:
                return self._search_semantic_only(query, limit)

            # Case 3: Hybrid search (both query and filters)
            if query and filters:
                return self._search_hybrid(query, filters, limit)

            # Case 4: No query and no filters — return empty
            return []

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search failed: {e}") from e

    def _search_structured_only(self, filters: Filters, limit: int) -> list[Result]:
        """Execute structured-only search using SQL filters."""
        docs = self.structured_store.query(
            type_filter=filters.data_types[0] if filters.data_types else None,
            contact=filters.contacts[0] if filters.contacts else None,
            after=filters.date_after,
            before=filters.date_before,
            metadata=filters.metadata,
            limit=limit,
        )

        # Convert to Results with score=1.0 for structured matches
        results = []
        for doc in docs:
            # Apply additional filters if present
            if self._matches_filters(doc, filters):
                results.append(
                    Result(
                        document=doc,
                        score=1.0,
                        match_type="structured",
                    )
                )

        return results[:limit]

    def _search_semantic_only(self, query: str, limit: int) -> list[Result]:
        """Execute semantic-only search using vector or FTS."""
        results = []

        # Try vector search first
        if self.vector_store:
            try:
                vector_results = self.vector_store.query(text=query, n=limit)
                for doc, score in vector_results:
                    results.append(
                        Result(
                            document=doc,
                            score=score,
                            match_type="semantic",
                        )
                    )
                return results
            except SearchError:
                logger.warning("Vector search failed, falling back to FTS")

        # Fall back to full-text search
        docs = self.structured_store.query(text_search=query, limit=limit)

        # Score based on match type (better if multiple words match)
        for doc in docs:
            # Simple scoring: count matching words
            query_words = set(query.lower().split())
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words & doc_words)
            score = min(1.0, overlap / max(len(query_words), 1))

            results.append(
                Result(
                    document=doc,
                    score=max(0.5, score),  # Floor at 0.5 for FTS matches
                    match_type="structured",  # FTS is structured search
                )
            )

        return results

    def _search_hybrid(
        self,
        query: str,
        filters: Filters,
        limit: int,
    ) -> list[Result]:
        """Execute hybrid search combining filters and semantic ranking."""
        # Start with structured query
        docs = self.structured_store.query(
            type_filter=filters.data_types[0] if filters.data_types else None,
            contact=filters.contacts[0] if filters.contacts else None,
            after=filters.date_after,
            before=filters.date_before,
            text_search=query,
            metadata=filters.metadata,
            limit=limit * 2,  # Get extra results to re-rank
        )

        # Filter and score
        results = []
        for doc in docs:
            if not self._matches_filters(doc, filters):
                continue

            # Try to get semantic score
            score = 0.5  # Default for FTS matches

            if self.vector_store:
                try:
                    vector_results = self.vector_store.query(
                        text=query,
                        where={"document_type": doc.type.value},
                        n=1,
                    )
                    # Find if this doc is in vector results
                    for vec_doc, vec_score in vector_results:
                        if vec_doc.id == doc.id:
                            score = vec_score
                            break
                except SearchError:
                    pass

            results.append(
                Result(
                    document=doc,
                    score=score,
                    match_type="hybrid" if score > 0.5 else "structured",
                )
            )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def get_by_id(self, document_id: str) -> Document | None:
        """Retrieve a specific document by ID.

        Args:
            document_id: The document ID to retrieve.

        Returns:
            The Document, or None if not found.

        Raises:
            SearchError: If retrieval fails.
        """
        try:
            return self.structured_store.get_by_id(document_id)
        except SearchError:
            raise
        except Exception as e:
            raise SearchError(f"Failed to get document by ID: {e}") from e

    def get_thread(
        self,
        thread_id: int,
        around_timestamp: datetime | None = None,
        window: int = 10,
    ) -> list[Document]:
        """Retrieve messages in a conversation thread.

        Args:
            thread_id: The conversation thread ID (chat_id in metadata).
            around_timestamp: Center the window around this time. None for latest.
            window: Number of messages to return.

        Returns:
            List of Documents in chronological order.

        Raises:
            SearchError: If retrieval fails.
        """
        try:
            # Get documents with thread_id
            docs = self.structured_store.query(limit=window * 10)

            # Filter to matching thread
            thread_docs = []
            for doc in docs:
                if doc.metadata.get("chat_id") == thread_id:
                    thread_docs.append(doc)

            # Sort by timestamp
            if around_timestamp:
                # Split documents into before/after
                before = [d for d in thread_docs if d.timestamp and d.timestamp <= around_timestamp]
                after = [d for d in thread_docs if d.timestamp and d.timestamp > around_timestamp]

                # Sort and combine
                before.sort(key=lambda d: d.timestamp, reverse=True)
                after.sort(key=lambda d: d.timestamp)

                # Take half from each side
                half_window = window // 2
                result = before[:half_window]
                result.extend(after[: window - half_window])
                result.sort(key=lambda d: d.timestamp or datetime.min)
                return result

            # No timestamp filter: just return latest messages in chronological order
            thread_docs.sort(key=lambda d: d.timestamp or datetime.min)
            return thread_docs[-window:] if len(thread_docs) > window else thread_docs

        except Exception as e:
            logger.error(f"Failed to get thread {thread_id}: {e}")
            raise SearchError(f"Failed to get thread: {e}") from e

    def _matches_filters(self, doc: Document, filters: Filters) -> bool:
        """Check if a document matches all filters.

        Args:
            doc: The document to check.
            filters: The filters to apply.

        Returns:
            True if the document matches all filters.
        """
        # Type filter
        if filters.data_types and doc.type not in filters.data_types:
            return False

        # Contact filter
        if filters.contacts:
            doc_contacts = [doc.metadata.get("handle")]
            if not any(h in filters.contacts for h in doc_contacts if h):
                return False

        # Date range filter
        if doc.timestamp:
            if filters.date_after and doc.timestamp < filters.date_after:
                return False
            if filters.date_before and doc.timestamp > filters.date_before:
                return False

        # Attachment filter
        if filters.has_attachments is not None:
            has_attachments = len(doc.attachments) > 0
            if has_attachments != filters.has_attachments:
                return False

        # Backup ID filter
        if filters.backup_id and doc.source.backup_id != filters.backup_id:
            return False

        # Thread ID filter
        if filters.thread_id and doc.metadata.get("chat_id") != filters.thread_id:
            return False

        # Metadata filter
        if filters.metadata:
            for key, value in filters.metadata.items():
                if doc.metadata.get(key) != value:
                    return False

        return True
