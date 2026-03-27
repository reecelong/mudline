"""SQLite-backed structured store with FTS5 full-text search.

This module provides persistent storage for Document objects with support for
structured filtering (type, contact, date range) and full-text search.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mudline.exceptions import SearchError
from mudline.models.document import Document, DocumentType

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StructuredStore:
    """SQLite-backed store for documents with FTS5 full-text search.

    The store maintains a normalized documents table with type, timestamp,
    contact_handle, thread_id, domain columns and supports arbitrary JSONB
    metadata. Full-text search is powered by SQLite FTS5.

    Args:
        db_path: Path to SQLite database file. Use ":memory:" for in-memory.
    """

    db_path: str | Path = ":memory:"
    _conn: sqlite3.Connection | None = None

    def __post_init__(self) -> None:
        """Initialize database schema on first use."""
        object.__setattr__(self, "db_path", str(self.db_path))
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory and foreign keys enabled.

        For in-memory databases, reuses the same connection to preserve schema.
        For file-based databases, creates a new connection each time.
        """
        if self.db_path == ":memory:":
            # Reuse connection for in-memory databases to preserve schema
            if self._conn is None:
                conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                object.__setattr__(self, "_conn", conn)
            return self._conn
        else:
            # Create new connections for file-based databases
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            return conn

    def _init_schema(self) -> None:
        """Create or verify database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Main documents table with normalized fields
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp DATETIME,
                    contact_handle TEXT,
                    thread_id INTEGER,
                    domain TEXT NOT NULL,
                    backup_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    backup_timestamp DATETIME NOT NULL,
                    metadata TEXT NOT NULL,
                    has_attachments BOOLEAN NOT NULL DEFAULT 0,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Indexes for common query patterns
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_type
                ON documents(type)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_contact
                ON documents(contact_handle)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_thread
                ON documents(thread_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_timestamp
                ON documents(timestamp)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_backup
                ON documents(backup_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_domain
                ON documents(domain)
                """
            )

            # FTS5 virtual table for full-text search
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
                USING fts5(
                    id UNINDEXED,
                    text,
                    contact_handle UNINDEXED,
                    type UNINDEXED
                )
                """
            )

            conn.commit()

    def insert(self, doc: Document) -> None:
        """Insert a single document into the store.

        Args:
            doc: Document to insert.

        Raises:
            SearchError: If insertion fails.
        """
        try:
            self.insert_batch([doc])
        except Exception as e:
            logger.error(f"Failed to insert document {doc.id}: {e}")
            raise SearchError(f"Failed to insert document: {e}") from e

    def insert_batch(self, docs: list[Document]) -> None:
        """Insert multiple documents efficiently.

        Handles deduplication by document ID — if a document with the same ID
        exists, it is replaced.

        Args:
            docs: List of documents to insert.

        Raises:
            SearchError: If batch insertion fails.
        """
        if not docs:
            return

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                for doc in docs:
                    # Extract contact_handle and thread_id from metadata
                    contact_handle = None
                    thread_id = None

                    if doc.type == DocumentType.MESSAGE:
                        contact_handle = doc.metadata.get("handle")
                        thread_id = doc.metadata.get("chat_id")
                    elif doc.type in (
                        DocumentType.CALL,
                        DocumentType.VOICEMAIL,
                    ):
                        contact_handle = doc.metadata.get("handle")
                    elif doc.type == DocumentType.CONTACT:
                        # For contacts, use email or first phone as handle
                        emails = doc.metadata.get("emails", [])
                        phones = doc.metadata.get("phones", [])
                        contact_handle = emails[0] if emails else (phones[0] if phones else None)

                    # Insert or replace in main documents table
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO documents (
                            id, type, text, timestamp, contact_handle, thread_id,
                            domain, backup_id, relative_path, backup_timestamp,
                            metadata, has_attachments
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc.id,
                            doc.type.value,
                            doc.text,
                            doc.timestamp.isoformat() if doc.timestamp else None,
                            contact_handle,
                            thread_id,
                            doc.source.domain,
                            doc.source.backup_id,
                            doc.source.relative_path,
                            doc.source.backup_timestamp.isoformat(),
                            json.dumps(doc.metadata),
                            bool(doc.attachments),
                        ),
                    )

                    # Insert into FTS table
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO documents_fts (id, text, contact_handle, type)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            doc.id,
                            doc.text,
                            contact_handle or "",
                            doc.type.value,
                        ),
                    )

                conn.commit()
                logger.info(f"Inserted {len(docs)} documents")

        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
            raise SearchError(f"Failed to insert batch: {e}") from e

    def query(
        self,
        type_filter: DocumentType | None = None,
        contact: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        text_search: str | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """Query documents with optional filters and full-text search.

        Supports any combination of filters. All filters are ANDed together.

        Args:
            type_filter: Filter by document type.
            contact: Filter by contact handle (phone/email).
            after: Filter to documents with timestamp >= this datetime.
            before: Filter to documents with timestamp <= this datetime.
            text_search: Full-text search query.
            limit: Maximum number of results to return.

        Returns:
            List of Document objects sorted by rank and timestamp (descending).

        Raises:
            SearchError: If query fails.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Build query based on whether we're doing FTS or just SQL
                if text_search:
                    results = self._fts_query(
                        cursor,
                        text_search,
                        type_filter,
                        contact,
                        after,
                        before,
                        limit,
                    )
                else:
                    results = self._sql_query(cursor, type_filter, contact, after, before, limit)

                return results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise SearchError(f"Query failed: {e}") from e

    def _fts_query(
        self,
        cursor: sqlite3.Cursor,
        text_search: str,
        type_filter: DocumentType | None = None,
        contact: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """Execute FTS query with optional SQL filters."""
        # Build WHERE clause for FTS
        fts_where = "documents_fts MATCH ?"
        fts_params: list[Any] = [text_search]

        # Join with documents table for filtering
        query = f"""
            SELECT d.*, rank FROM (
                SELECT id, rank FROM documents_fts WHERE {fts_where}
            ) AS fts
            JOIN documents AS d ON fts.id = d.id
            WHERE 1=1
        """

        sql_params = fts_params.copy()

        # Add SQL filters
        if type_filter:
            query += " AND d.type = ?"
            sql_params.append(type_filter.value)

        if contact:
            query += " AND d.contact_handle = ?"
            sql_params.append(contact)

        if after:
            query += " AND d.timestamp >= ?"
            sql_params.append(after.isoformat())

        if before:
            query += " AND d.timestamp <= ?"
            sql_params.append(before.isoformat())

        query += " ORDER BY rank ASC, d.timestamp DESC LIMIT ?"
        sql_params.append(limit)

        cursor.execute(query, sql_params)
        rows = cursor.fetchall()

        return [self._row_to_document(row) for row in rows]

    def _sql_query(
        self,
        cursor: sqlite3.Cursor,
        type_filter: DocumentType | None = None,
        contact: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """Execute pure SQL query (no FTS)."""
        query = "SELECT * FROM documents WHERE 1=1"
        params: list[Any] = []

        if type_filter:
            query += " AND type = ?"
            params.append(type_filter.value)

        if contact:
            query += " AND contact_handle = ?"
            params.append(contact)

        if after:
            query += " AND timestamp >= ?"
            params.append(after.isoformat())

        if before:
            query += " AND timestamp <= ?"
            params.append(before.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_document(row) for row in rows]

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """Convert a database row back to a Document object."""
        from mudline.models.document import Source

        metadata = json.loads(row["metadata"])

        timestamp = None
        if row["timestamp"]:
            timestamp = datetime.fromisoformat(row["timestamp"])

        backup_timestamp = datetime.fromisoformat(row["backup_timestamp"])

        # Note: we don't store attachments in the database for now
        # They would need to be re-resolved from the backup
        source = Source(
            backup_id=row["backup_id"],
            domain=row["domain"],
            relative_path=row["relative_path"],
            backup_timestamp=backup_timestamp,
        )

        doc = Document(
            id=row["id"],
            type=DocumentType(row["type"]),
            text=row["text"],
            source=source,
            timestamp=timestamp,
            metadata=metadata,
            attachments=[],  # Attachments would need to be re-resolved
        )

        return doc

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
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
                row = cursor.fetchone()

                if not row:
                    return None

                return self._row_to_document(row)

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise SearchError(f"Failed to get document: {e}") from e

    def count(self) -> int:
        """Return total number of documents in store.

        Returns:
            Count of documents.

        Raises:
            SearchError: If count query fails.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents")
                return cursor.fetchone()[0]  # type: ignore
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            raise SearchError(f"Failed to count documents: {e}") from e

    def delete_all(self) -> None:
        """Delete all documents from the store.

        Used for testing and cleanup.

        Raises:
            SearchError: If deletion fails.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents_fts")
                cursor.execute("DELETE FROM documents")
                conn.commit()
                logger.info("Deleted all documents")
        except Exception as e:
            logger.error(f"Failed to delete all documents: {e}")
            raise SearchError(f"Failed to delete all documents: {e}") from e
