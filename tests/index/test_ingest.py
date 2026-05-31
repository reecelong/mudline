"""Tests for the ingest pipeline."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from mudline.exceptions import SearchError
from mudline.index.ingest import IngestPipeline, IngestState
from mudline.index.structured import StructuredStore
from mudline.models.document import Document, DocumentType, Source


@pytest.fixture
def structured_store() -> StructuredStore:
    """Create an in-memory structured store for testing."""
    return StructuredStore(db_path=":memory:")


@pytest.fixture
def sample_source() -> Source:
    """Create a sample Source for testing."""
    return Source(
        backup_id="test-backup-123",
        domain="HomeDomain",
        relative_path="Library/SMS/sms.db",
        backup_timestamp=datetime.now(),
    )


@pytest.fixture
def sample_documents(sample_source: Source) -> list[Document]:
    """Create sample documents for testing."""
    docs = []
    for i in range(5):
        docs.append(
            Document(
                type=DocumentType.MESSAGE,
                text=f"Message {i}: Hello, how are you?",
                source=sample_source,
                timestamp=datetime.now(),
                metadata={
                    "handle": "+15551234567",
                    "chat_id": 1,
                    "is_from_me": False,
                },
            )
        )
    return docs


class TestIngestState:
    """Test the IngestState dataclass."""

    def test_create_ingest_state(self) -> None:
        """Test creating an IngestState."""
        now = datetime.now()
        state = IngestState(
            backup_id="test-backup",
            started_at=now,
            status="in_progress",
        )

        assert state.backup_id == "test-backup"
        assert state.started_at == now
        assert state.completed_at is None
        assert state.document_count == 0
        assert state.status == "in_progress"

    def test_completed_state(self) -> None:
        """Test creating a completed IngestState."""
        now = datetime.now()
        later = datetime.now()
        state = IngestState(
            backup_id="test-backup",
            started_at=now,
            completed_at=later,
            document_count=100,
            status="completed",
        )

        assert state.status == "completed"
        assert state.document_count == 100


class TestIngestPipeline:
    """Test the IngestPipeline class."""

    def test_initialization(self, structured_store: StructuredStore) -> None:
        """Test pipeline initialization."""
        pipeline = IngestPipeline(structured_store)

        assert pipeline.structured_store == structured_store
        assert pipeline.vector_store is None
        assert pipeline.batch_size == 100

    def test_initialization_with_vector_store(self, structured_store: StructuredStore) -> None:
        """Test pipeline initialization with vector store."""
        mock_vector_store = MagicMock()
        pipeline = IngestPipeline(structured_store, mock_vector_store)

        assert pipeline.vector_store == mock_vector_store

    def test_ingest_simple(
        self,
        structured_store: StructuredStore,
        sample_documents: list[Document],
    ) -> None:
        """Test basic ingest operation."""
        pipeline = IngestPipeline(structured_store)

        state = pipeline.ingest(
            backup_id="test-backup",
            documents=iter(sample_documents),
        )

        assert state.backup_id == "test-backup"
        assert state.status == "completed"
        assert state.document_count == len(sample_documents)
        assert state.completed_at is not None

    def test_ingest_with_batching(
        self,
        structured_store: StructuredStore,
        sample_source: Source,
    ) -> None:
        """Test that ingest properly batches documents."""
        pipeline = IngestPipeline(structured_store)
        pipeline.batch_size = 2

        # Create more documents than batch size
        docs = [
            Document(
                type=DocumentType.MESSAGE,
                text=f"Message {i}",
                source=sample_source,
                timestamp=datetime.now(),
                metadata={"handle": "+15551234567", "chat_id": 1},
            )
            for i in range(5)
        ]

        state = pipeline.ingest(
            backup_id="test-backup",
            documents=iter(docs),
        )

        assert state.document_count == 5
        assert structured_store.count() == 5

    def test_ingest_with_progress_callback(
        self,
        structured_store: StructuredStore,
        sample_documents: list[Document],
    ) -> None:
        """Test that progress callback is called."""
        pipeline = IngestPipeline(structured_store)

        progress_calls = []

        def on_progress(count: int) -> None:
            progress_calls.append(count)

        pipeline.ingest(
            backup_id="test-backup",
            documents=iter(sample_documents),
            on_progress=on_progress,
        )

        # Should have been called at least once
        assert len(progress_calls) > 0

    def test_is_ingested_false_initially(self, structured_store: StructuredStore) -> None:
        """Test that is_ingested returns False for new backup."""
        pipeline = IngestPipeline(structured_store)

        assert not pipeline.is_ingested("nonexistent-backup")

    def test_is_ingested_true_after_ingest(
        self,
        structured_store: StructuredStore,
        sample_documents: list[Document],
    ) -> None:
        """Test that is_ingested returns True after successful ingest."""
        pipeline = IngestPipeline(structured_store)

        pipeline.ingest("test-backup", iter(sample_documents))

        assert pipeline.is_ingested("test-backup")

    def test_get_state_after_ingest(
        self,
        structured_store: StructuredStore,
        sample_documents: list[Document],
    ) -> None:
        """Test retrieving state after ingest."""
        pipeline = IngestPipeline(structured_store)

        original_state = pipeline.ingest("test-backup", iter(sample_documents))

        retrieved_state = pipeline.get_state("test-backup")

        assert retrieved_state is not None
        assert retrieved_state.backup_id == original_state.backup_id
        assert retrieved_state.document_count == original_state.document_count
        assert retrieved_state.status == original_state.status

    def test_get_state_nonexistent(self, structured_store: StructuredStore) -> None:
        """Test getting state for nonexistent backup."""
        pipeline = IngestPipeline(structured_store)

        state = pipeline.get_state("nonexistent-backup")

        assert state is None

    def test_idempotent_ingest(
        self,
        structured_store: StructuredStore,
        sample_documents: list[Document],
    ) -> None:
        """Test that re-ingesting same backup is idempotent."""
        pipeline = IngestPipeline(structured_store)

        # First ingest
        state1 = pipeline.ingest("test-backup", iter(sample_documents))

        # Second ingest should be skipped
        state2 = pipeline.ingest("test-backup", iter(sample_documents))

        assert state1.document_count == state2.document_count
        assert structured_store.count() == len(sample_documents)

    def test_ingest_with_empty_documents(
        self,
        structured_store: StructuredStore,
    ) -> None:
        """Test ingesting empty document list."""
        pipeline = IngestPipeline(structured_store)

        state = pipeline.ingest("test-backup", iter([]))

        assert state.document_count == 0
        assert state.status == "completed"

    def test_ingest_with_vector_store_calls_add(
        self,
        structured_store: StructuredStore,
        sample_documents: list[Document],
    ) -> None:
        """Test that vector store is called during ingest."""
        mock_vector_store = MagicMock()
        pipeline = IngestPipeline(structured_store, mock_vector_store)

        pipeline.ingest("test-backup", iter(sample_documents))

        # Vector store add should have been called
        assert mock_vector_store.add.called

    def test_ingest_failure_records_failed_state(
        self,
        structured_store: StructuredStore,
        sample_documents: list[Document],
    ) -> None:
        """Test that failed ingest records failed state."""
        pipeline = IngestPipeline(structured_store)

        # Patch insert_batch to fail
        with (
            patch.object(structured_store, "insert_batch", side_effect=Exception("Test error")),
            pytest.raises(SearchError),
        ):
            pipeline.ingest("test-backup", iter(sample_documents))

        # State should be recorded as failed
        state = pipeline.get_state("test-backup")
        assert state is not None
        assert state.status == "failed"

    def test_ingest_state_timestamps(
        self,
        structured_store: StructuredStore,
        sample_documents: list[Document],
    ) -> None:
        """Test that ingest state has correct timestamps."""
        pipeline = IngestPipeline(structured_store)

        before_ingest = datetime.now()
        state = pipeline.ingest("test-backup", iter(sample_documents))
        after_ingest = datetime.now()

        assert before_ingest <= state.started_at <= after_ingest
        assert state.completed_at is not None
        assert state.started_at <= state.completed_at <= after_ingest

    def test_ingest_different_backup_ids(
        self,
        structured_store: StructuredStore,
        sample_documents: list[Document],
        sample_source: Source,
    ) -> None:
        """Test ingesting documents from different backups."""
        pipeline = IngestPipeline(structured_store)

        # Ingest first backup
        pipeline.ingest("backup-1", iter(sample_documents))

        # Create documents with different backup_id
        source2 = Source(
            backup_id="backup-2",
            domain=sample_source.domain,
            relative_path=sample_source.relative_path,
            backup_timestamp=datetime.now(),
        )
        docs2 = [
            Document(
                type=DocumentType.MESSAGE,
                text=f"Message {i}",
                source=source2,
                timestamp=datetime.now(),
                metadata={"handle": "+15559876543", "chat_id": 2},
            )
            for i in range(3)
        ]

        pipeline.ingest("backup-2", iter(docs2))

        # Both should be marked as ingested
        assert pipeline.is_ingested("backup-1")
        assert pipeline.is_ingested("backup-2")
