"""Tests for the hybrid retriever."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from mudline.exceptions import SearchError
from mudline.index.retriever import HybridRetriever
from mudline.index.structured import StructuredStore
from mudline.models.document import Document, DocumentType, Source
from mudline.models.retriever import Filters


@pytest.fixture
def structured_store() -> StructuredStore:
    """Create an in-memory structured store with sample documents."""
    store = StructuredStore(db_path=":memory:")

    # Insert sample documents
    source = Source(
        backup_id="test-backup",
        domain="HomeDomain",
        relative_path="Library/SMS/sms.db",
        backup_timestamp=datetime.now(),
    )

    base_time = datetime.now()
    docs = [
        Document(
            type=DocumentType.MESSAGE,
            text="Let's meet for coffee at the cafe",
            source=source,
            timestamp=base_time,
            metadata={
                "handle": "+15551234567",
                "chat_id": 1,
                "is_from_me": False,
            },
        ),
        Document(
            type=DocumentType.MESSAGE,
            text="Sure, I'll see you at 3pm",
            source=source,
            timestamp=base_time + timedelta(minutes=5),
            metadata={
                "handle": "+15551234567",
                "chat_id": 1,
                "is_from_me": True,
            },
        ),
        Document(
            type=DocumentType.MESSAGE,
            text="What about the project deadline?",
            source=source,
            timestamp=base_time + timedelta(minutes=10),
            metadata={
                "handle": "+15559876543",
                "chat_id": 2,
                "is_from_me": False,
            },
        ),
        Document(
            type=DocumentType.NOTE,
            text="Project notes: focus on architecture design",
            source=source,
            timestamp=base_time + timedelta(minutes=15),
            metadata={"folder": "Work"},
        ),
    ]

    store.insert_batch(docs)
    return store


class TestHybridRetriever:
    """Test the HybridRetriever class."""

    def test_initialization(self, structured_store: StructuredStore) -> None:
        """Test retriever initialization."""
        retriever = HybridRetriever(structured_store)

        assert retriever.structured_store == structured_store
        assert retriever.vector_store is None

    def test_initialization_with_vector_store(self, structured_store: StructuredStore) -> None:
        """Test retriever initialization with vector store."""
        mock_vector_store = MagicMock()
        retriever = HybridRetriever(structured_store, mock_vector_store)

        assert retriever.vector_store == mock_vector_store

    def test_search_structured_only(self, structured_store: StructuredStore) -> None:
        """Test structured-only search (filters, no query)."""
        retriever = HybridRetriever(structured_store)

        filters = Filters(data_types=[DocumentType.MESSAGE])
        results = retriever.search(filters=filters, limit=10)

        assert len(results) == 3  # 3 messages in store
        assert all(r.document.type == DocumentType.MESSAGE for r in results)
        assert all(r.score == 1.0 for r in results)
        assert all(r.match_type == "structured" for r in results)

    def test_search_semantic_only_falls_back_to_fts(
        self, structured_store: StructuredStore
    ) -> None:
        """Test semantic-only search falls back to FTS when no vector store."""
        retriever = HybridRetriever(structured_store)

        results = retriever.search(query="coffee", limit=10)

        assert len(results) > 0
        # Should find the coffee message
        assert any("coffee" in r.document.text.lower() for r in results)

    def test_search_by_contact(self, structured_store: StructuredStore) -> None:
        """Test filtering by contact."""
        retriever = HybridRetriever(structured_store)

        filters = Filters(contacts=["+15551234567"])
        results = retriever.search(filters=filters, limit=10)

        assert len(results) == 2  # 2 messages from this contact
        assert all(r.document.metadata.get("handle") == "+15551234567" for r in results)

    def test_search_by_date_range(self, structured_store: StructuredStore) -> None:
        """Test filtering by date range."""
        retriever = HybridRetriever(structured_store)

        now = datetime.now()
        filters = Filters(
            date_after=now,
            date_before=now + timedelta(minutes=6),
        )
        results = retriever.search(filters=filters, limit=10)

        # Should only get the first two messages
        assert len(results) <= 2

    def test_search_by_type(self, structured_store: StructuredStore) -> None:
        """Test filtering by document type."""
        retriever = HybridRetriever(structured_store)

        filters = Filters(data_types=[DocumentType.NOTE])
        results = retriever.search(filters=filters, limit=10)

        assert len(results) == 1
        assert results[0].document.type == DocumentType.NOTE

    def test_search_empty_query_and_filters(self, structured_store: StructuredStore) -> None:
        """Test search with both query and filters empty returns empty."""
        retriever = HybridRetriever(structured_store)

        results = retriever.search(query=None, filters=None, limit=10)

        assert results == []

    def test_search_limit(self, structured_store: StructuredStore) -> None:
        """Test that search respects limit."""
        retriever = HybridRetriever(structured_store)

        filters = Filters(data_types=[DocumentType.MESSAGE])
        results = retriever.search(filters=filters, limit=1)

        assert len(results) == 1

    def test_get_by_id(self, structured_store: StructuredStore) -> None:
        """Test getting a document by ID."""
        retriever = HybridRetriever(structured_store)

        # Get a document ID from store
        docs = structured_store.query(limit=1)
        if docs:
            doc_id = docs[0].id

            retrieved = retriever.get_by_id(doc_id)

            assert retrieved is not None
            assert retrieved.id == doc_id

    def test_get_by_id_nonexistent(self, structured_store: StructuredStore) -> None:
        """Test getting a nonexistent document returns None."""
        retriever = HybridRetriever(structured_store)

        retrieved = retriever.get_by_id("nonexistent-id")

        assert retrieved is None

    def test_get_thread(self, structured_store: StructuredStore) -> None:
        """Test retrieving messages in a conversation thread."""
        retriever = HybridRetriever(structured_store)

        messages = retriever.get_thread(thread_id=1)

        assert len(messages) > 0
        assert all(m.metadata.get("chat_id") == 1 for m in messages)

    def test_get_thread_nonexistent(self, structured_store: StructuredStore) -> None:
        """Test getting nonexistent thread returns empty list."""
        retriever = HybridRetriever(structured_store)

        messages = retriever.get_thread(thread_id=999)

        assert messages == []

    def test_get_thread_with_window(self, structured_store: StructuredStore) -> None:
        """Test that get_thread respects window size."""
        retriever = HybridRetriever(structured_store)

        messages = retriever.get_thread(thread_id=1, window=1)

        assert len(messages) <= 1

    def test_get_thread_chronological_order(self, structured_store: StructuredStore) -> None:
        """Test that thread messages are in chronological order."""
        retriever = HybridRetriever(structured_store)

        messages = retriever.get_thread(thread_id=1)

        # Check timestamps are in order
        timestamps = [m.timestamp for m in messages if m.timestamp]
        assert timestamps == sorted(timestamps)

    def test_search_with_vector_store_reranking(self, structured_store: StructuredStore) -> None:
        """Test hybrid search with vector store re-ranking."""
        mock_vector_store = MagicMock()

        # Mock vector search results
        mock_doc = Document(
            type=DocumentType.MESSAGE,
            text="coffee",
            source=Source(
                backup_id="test",
                domain="test",
                relative_path="test",
                backup_timestamp=datetime.now(),
            ),
        )
        mock_vector_store.query.return_value = [(mock_doc, 0.9)]

        retriever = HybridRetriever(structured_store, mock_vector_store)

        filters = Filters(data_types=[DocumentType.MESSAGE])
        results = retriever.search(query="coffee", filters=filters, limit=10)

        assert len(results) > 0

    def test_search_vector_fallback_on_error(self, structured_store: StructuredStore) -> None:
        """Test that search falls back to FTS if vector search errors."""
        mock_vector_store = MagicMock()
        mock_vector_store.query.side_effect = SearchError("Vector search failed")

        retriever = HybridRetriever(structured_store, mock_vector_store)

        # Should still work, falling back to FTS
        results = retriever.search(query="coffee", limit=10)

        assert len(results) > 0

    def test_matches_filters_type(self, structured_store: StructuredStore) -> None:
        """Test filter matching by type."""
        retriever = HybridRetriever(structured_store)
        doc = Document(
            type=DocumentType.MESSAGE,
            text="test",
            source=Source(
                backup_id="test",
                domain="test",
                relative_path="test",
                backup_timestamp=datetime.now(),
            ),
        )

        filters = Filters(data_types=[DocumentType.MESSAGE])
        assert retriever._matches_filters(doc, filters)

        filters = Filters(data_types=[DocumentType.NOTE])
        assert not retriever._matches_filters(doc, filters)

    def test_matches_filters_contact(self, structured_store: StructuredStore) -> None:
        """Test filter matching by contact."""
        retriever = HybridRetriever(structured_store)
        doc = Document(
            type=DocumentType.MESSAGE,
            text="test",
            source=Source(
                backup_id="test",
                domain="test",
                relative_path="test",
                backup_timestamp=datetime.now(),
            ),
            metadata={"handle": "+15551234567"},
        )

        filters = Filters(contacts=["+15551234567"])
        assert retriever._matches_filters(doc, filters)

        filters = Filters(contacts=["+15559999999"])
        assert not retriever._matches_filters(doc, filters)

    def test_matches_filters_date_range(self, structured_store: StructuredStore) -> None:
        """Test filter matching by date range."""
        retriever = HybridRetriever(structured_store)

        now = datetime.now()
        doc = Document(
            type=DocumentType.MESSAGE,
            text="test",
            source=Source(
                backup_id="test",
                domain="test",
                relative_path="test",
                backup_timestamp=now,
            ),
            timestamp=now,
        )

        filters = Filters(date_after=now - timedelta(hours=1), date_before=now + timedelta(hours=1))
        assert retriever._matches_filters(doc, filters)

        filters = Filters(date_after=now + timedelta(hours=1))
        assert not retriever._matches_filters(doc, filters)

    def test_matches_filters_attachments(self, structured_store: StructuredStore) -> None:
        """Test filter matching by attachments."""
        retriever = HybridRetriever(structured_store)

        from mudline.models.document import Attachment

        doc = Document(
            type=DocumentType.MESSAGE,
            text="test",
            source=Source(
                backup_id="test",
                domain="test",
                relative_path="test",
                backup_timestamp=datetime.now(),
            ),
            attachments=[Attachment(filename="test.jpg", mime_type="image/jpeg")],
        )

        filters = Filters(has_attachments=True)
        assert retriever._matches_filters(doc, filters)

        filters = Filters(has_attachments=False)
        assert not retriever._matches_filters(doc, filters)

    def test_matches_filters_backup_id(self, structured_store: StructuredStore) -> None:
        """Test filter matching by backup ID."""
        retriever = HybridRetriever(structured_store)
        doc = Document(
            type=DocumentType.MESSAGE,
            text="test",
            source=Source(
                backup_id="backup-123",
                domain="test",
                relative_path="test",
                backup_timestamp=datetime.now(),
            ),
        )

        filters = Filters(backup_id="backup-123")
        assert retriever._matches_filters(doc, filters)

        filters = Filters(backup_id="backup-456")
        assert not retriever._matches_filters(doc, filters)

    def test_matches_filters_metadata(self, structured_store: StructuredStore) -> None:
        """Test filter matching by metadata."""
        retriever = HybridRetriever(structured_store)
        doc = Document(
            type=DocumentType.MESSAGE,
            text="test",
            source=Source(
                backup_id="test",
                domain="test",
                relative_path="test",
                backup_timestamp=datetime.now(),
            ),
            metadata={"custom_key": "custom_value"},
        )

        filters = Filters(metadata={"custom_key": "custom_value"})
        assert retriever._matches_filters(doc, filters)

        filters = Filters(metadata={"custom_key": "other_value"})
        assert not retriever._matches_filters(doc, filters)
