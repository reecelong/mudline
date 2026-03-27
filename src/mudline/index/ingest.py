"""Ingest pipeline for consuming extracted documents and indexing them.

This module provides the IngestPipeline class that takes extracted documents
and stores them in both structured and vector stores with progress tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from mudline.exceptions import SearchError

logger = logging.getLogger(__name__)


@dataclass
class IngestState:
    """State tracking for an ingest operation.

    Args:
        backup_id: The backup ID being ingested.
        started_at: When the ingest operation started.
        completed_at: When the ingest operation completed. None if in progress.
        document_count: Number of documents ingested.
        status: Current status ("in_progress", "completed", "failed").
    """

    backup_id: str
    started_at: datetime
    completed_at: datetime | None = None
    document_count: int = 0
    status: str = "in_progress"


class IngestPipeline:
    """Pipeline for ingesting extracted documents into storage.

    Consumes an iterator of Document objects from extractors and writes them
    to both structured (SQLite) and vector (ChromaDB) stores with batching
    and progress tracking.

    Args:
        structured_store: The StructuredStore for SQL-based storage.
        vector_store: Optional VectorStore for semantic search. If None, only
                      structured storage is used (graceful degradation).
    """

    def __init__(
        self,
        structured_store,  # type: StructuredStore
        vector_store=None,  # type: VectorStore | None
    ) -> None:
        """Initialize the ingest pipeline.

        Args:
            structured_store: Required SQLite-backed store.
            vector_store: Optional vector store for embeddings.
        """
        self.structured_store = structured_store
        self.vector_store = vector_store
        self.batch_size = 100
        self._init_ingest_state_table()

    def _init_ingest_state_table(self) -> None:
        """Initialize the ingest state tracking table."""
        try:
            conn = self.structured_store._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS _ingest_state (
                    backup_id TEXT PRIMARY KEY,
                    started_at DATETIME NOT NULL,
                    completed_at DATETIME,
                    document_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'in_progress'
                )
                """
            )

            conn.commit()
            logger.debug("Initialized ingest state table")

        except Exception as e:
            logger.error(f"Failed to initialize ingest state table: {e}")
            raise SearchError(f"Failed to initialize ingest state table: {e}") from e

    def ingest(
        self,
        backup_id: str,
        documents,  # type: Iterator[Document]
        on_progress=None,  # type: Callable[[int], None] | None
    ) -> IngestState:
        """Ingest extracted documents into storage.

        Batches documents and writes them to both stores with progress tracking.
        Re-ingestion of the same backup_id is idempotent (skips if already complete).

        Args:
            backup_id: Unique identifier for the backup.
            documents: Iterator of Document objects from extractors.
            on_progress: Optional callback called with batch count after each batch.

        Returns:
            IngestState with final status and document count.

        Raises:
            SearchError: If ingestion fails.
        """
        # Check if already ingested
        if self.is_ingested(backup_id):
            logger.info(f"Backup {backup_id} already ingested, skipping")
            state = self.get_state(backup_id)
            if state:
                return state
            raise SearchError(f"Backup {backup_id} state not found despite being marked ingested")

        try:
            start_time = datetime.now()
            self._record_ingest_state(
                backup_id,
                IngestState(
                    backup_id=backup_id,
                    started_at=start_time,
                    status="in_progress",
                ),
            )

            batch: list = []
            total_count = 0

            for doc in documents:
                batch.append(doc)

                if len(batch) >= self.batch_size:
                    # Insert batch
                    self.structured_store.insert_batch(batch)
                    if self.vector_store:
                        self.vector_store.add(batch)

                    total_count += len(batch)
                    if on_progress:
                        on_progress(total_count)

                    logger.debug(f"Ingested batch of {len(batch)} documents")
                    batch = []

            # Insert remaining documents
            if batch:
                self.structured_store.insert_batch(batch)
                if self.vector_store:
                    self.vector_store.add(batch)

                total_count += len(batch)
                if on_progress:
                    on_progress(total_count)

                logger.debug(f"Ingested final batch of {len(batch)} documents")

            # Mark as completed
            completed_state = IngestState(
                backup_id=backup_id,
                started_at=start_time,
                completed_at=datetime.now(),
                document_count=total_count,
                status="completed",
            )

            self._record_ingest_state(backup_id, completed_state)

            logger.info(f"Completed ingestion of {total_count} documents for {backup_id}")
            return completed_state

        except Exception as e:
            logger.error(f"Ingest failed for {backup_id}: {e}")
            # Mark as failed
            failed_state = IngestState(
                backup_id=backup_id,
                started_at=start_time,
                completed_at=datetime.now(),
                status="failed",
            )
            self._record_ingest_state(backup_id, failed_state)
            raise SearchError(f"Ingest failed for {backup_id}: {e}") from e

    def is_ingested(self, backup_id: str) -> bool:
        """Check if a backup has been successfully ingested.

        Args:
            backup_id: The backup ID to check.

        Returns:
            True if the backup has been successfully ingested.
        """
        try:
            state = self.get_state(backup_id)
            return state is not None and state.status == "completed"
        except Exception as e:
            logger.error(f"Failed to check ingest status for {backup_id}: {e}")
            return False

    def get_state(self, backup_id: str) -> IngestState | None:
        """Get the ingest state for a backup.

        Args:
            backup_id: The backup ID to look up.

        Returns:
            The IngestState, or None if not found.

        Raises:
            SearchError: If the query fails.
        """
        try:
            conn = self.structured_store._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM _ingest_state WHERE backup_id = ?",
                (backup_id,),
            )

            row = cursor.fetchone()

            if not row:
                return None

            # Convert row to dict
            state_dict = {
                "backup_id": row[0],
                "started_at": datetime.fromisoformat(row[1]),
                "completed_at": datetime.fromisoformat(row[2]) if row[2] else None,
                "document_count": row[3],
                "status": row[4],
            }

            return IngestState(**state_dict)

        except Exception as e:
            logger.error(f"Failed to get ingest state for {backup_id}: {e}")
            raise SearchError(f"Failed to get ingest state: {e}") from e

    def _record_ingest_state(self, backup_id: str, state: IngestState) -> None:
        """Record ingest state in the database.

        Args:
            backup_id: The backup ID.
            state: The IngestState to record.

        Raises:
            SearchError: If recording fails.
        """
        try:
            conn = self.structured_store._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO _ingest_state
                (backup_id, started_at, completed_at, document_count, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    state.backup_id,
                    state.started_at.isoformat(),
                    state.completed_at.isoformat() if state.completed_at else None,
                    state.document_count,
                    state.status,
                ),
            )

            conn.commit()

        except Exception as e:
            logger.error(f"Failed to record ingest state: {e}")
            raise SearchError(f"Failed to record ingest state: {e}") from e
