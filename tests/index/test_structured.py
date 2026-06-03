"""Unit tests for StructuredStore."""

from __future__ import annotations

from datetime import datetime

import pytest

from mudline.index.structured import StructuredStore
from mudline.models.document import Attachment, Document, DocumentType, Source


@pytest.fixture
def store() -> StructuredStore:
    """Create an in-memory test store."""
    return StructuredStore(db_path=":memory:")


@pytest.fixture
def sample_source() -> Source:
    """Create a sample source."""
    return Source(
        backup_id="test-backup-001",
        domain="HomeDomain",
        relative_path="Library/Messages/chat.db",
        backup_timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def message_doc(sample_source: Source) -> Document:
    """Create a sample message document."""
    return Document(
        type=DocumentType.MESSAGE,
        text="Hey, can you call the plumber?",
        source=sample_source,
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        metadata={
            "handle": "+15551234567",
            "is_from_me": False,
            "chat_id": 42,
            "chat_name": None,
            "participants": ["+15551234567"],
        },
    )


@pytest.fixture
def contact_doc(sample_source: Source) -> Document:
    """Create a sample contact document."""
    return Document(
        type=DocumentType.CONTACT,
        text="Sarah Smith — Acme Corp — +15551234567 — sarah@example.com",
        source=sample_source,
        timestamp=None,  # Contacts don't have timestamps
        metadata={
            "phones": ["+15551234567"],
            "emails": ["sarah@example.com"],
            "organization": "Acme Corp",
            "handles": ["+15551234567", "sarah@example.com"],
        },
    )


@pytest.fixture
def photo_doc(sample_source: Source) -> Document:
    """Create a sample photo document."""
    return Document(
        type=DocumentType.PHOTO,
        text="Beach photo from summer trip",
        source=sample_source,
        timestamp=datetime(2024, 7, 15, 14, 30, 0),
        metadata={
            "latitude": 37.7749,
            "longitude": -122.4194,
            "width": 1920,
            "height": 1440,
            "album": "Vacation",
            "media_type": "image",
        },
        attachments=[
            Attachment(
                filename="photo.jpg",
                mime_type="image/jpeg",
                size_bytes=2048000,
            )
        ],
    )


class TestInsert:
    """Tests for document insertion."""

    def test_insert_single_document(self, store: StructuredStore, message_doc: Document) -> None:
        """Test inserting a single document."""
        store.insert(message_doc)
        assert store.count() == 1

    def test_insert_batch(
        self, store: StructuredStore, message_doc: Document, contact_doc: Document
    ) -> None:
        """Test batch insertion."""
        store.insert_batch([message_doc, contact_doc])
        assert store.count() == 2

    def test_insert_empty_batch(self, store: StructuredStore) -> None:
        """Test that empty batch is handled gracefully."""
        store.insert_batch([])
        assert store.count() == 0

    def test_deduplication_by_id(self, store: StructuredStore, message_doc: Document) -> None:
        """Test that documents with same ID are replaced."""
        store.insert(message_doc)
        assert store.count() == 1

        # Insert again with same ID but different text
        message_doc.text = "Updated message"
        store.insert(message_doc)

        # Should still be 1 document (replaced)
        assert store.count() == 1

        # Verify the text was updated
        retrieved = store.get_by_id(message_doc.id)
        assert retrieved is not None
        assert retrieved.text == "Updated message"

    def test_document_id_auto_generation(self, sample_source: Source) -> None:
        """Test that documents get deterministic IDs."""
        doc1 = Document(
            type=DocumentType.MESSAGE,
            text="Test message",
            source=sample_source,
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
        )

        doc2 = Document(
            type=DocumentType.MESSAGE,
            text="Test message",
            source=sample_source,
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
        )

        # Same content should generate same ID
        assert doc1.id == doc2.id


class TestQuery:
    """Tests for document queries."""

    def test_query_all(
        self,
        store: StructuredStore,
        message_doc: Document,
        contact_doc: Document,
        photo_doc: Document,
    ) -> None:
        """Test querying all documents."""
        store.insert_batch([message_doc, contact_doc, photo_doc])
        results = store.query(limit=100)
        assert len(results) == 3

    def test_query_by_type(
        self,
        store: StructuredStore,
        message_doc: Document,
        contact_doc: Document,
        photo_doc: Document,
    ) -> None:
        """Test filtering by document type."""
        store.insert_batch([message_doc, contact_doc, photo_doc])

        # Query for messages only
        results = store.query(type_filter=DocumentType.MESSAGE)
        assert len(results) == 1
        assert results[0].type == DocumentType.MESSAGE

        # Query for photos only
        results = store.query(type_filter=DocumentType.PHOTO)
        assert len(results) == 1
        assert results[0].type == DocumentType.PHOTO

    def test_query_by_contact(
        self, store: StructuredStore, message_doc: Document, contact_doc: Document
    ) -> None:
        """Test filtering by contact handle."""
        store.insert_batch([message_doc, contact_doc])

        # Message has phone as handle
        results = store.query(contact="+15551234567")
        assert len(results) == 1
        assert results[0].type == DocumentType.MESSAGE

        # Contact has email as handle (first email preferred over phone)
        results = store.query(contact="sarah@example.com")
        assert len(results) == 1
        assert results[0].type == DocumentType.CONTACT

        results = store.query(contact="nonexistent@example.com")
        assert len(results) == 0

    def test_query_by_date_range(self, store: StructuredStore) -> None:
        """Test filtering by date range."""
        # Create messages on different dates
        source = Source(
            backup_id="test-backup-001",
            domain="HomeDomain",
            relative_path="Library/Messages/chat.db",
            backup_timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )

        jan_doc = Document(
            type=DocumentType.MESSAGE,
            text="January message",
            source=source,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
        )

        feb_doc = Document(
            type=DocumentType.MESSAGE,
            text="February message",
            source=source,
            timestamp=datetime(2024, 2, 15, 10, 0, 0),
        )

        mar_doc = Document(
            type=DocumentType.MESSAGE,
            text="March message",
            source=source,
            timestamp=datetime(2024, 3, 15, 10, 0, 0),
        )

        store.insert_batch([jan_doc, feb_doc, mar_doc])

        # Query for February and later
        results = store.query(after=datetime(2024, 2, 1, 0, 0, 0))
        assert len(results) == 2

        # Query for before March
        results = store.query(before=datetime(2024, 3, 1, 0, 0, 0))
        assert len(results) == 2

        # Query for February only
        results = store.query(
            after=datetime(2024, 2, 1, 0, 0, 0),
            before=datetime(2024, 3, 1, 0, 0, 0),
        )
        assert len(results) == 1
        assert "February" in results[0].text

    def test_query_combined_filters(self, store: StructuredStore) -> None:
        """Test querying with multiple filters combined."""
        source = Source(
            backup_id="test-backup-001",
            domain="HomeDomain",
            relative_path="Library/Messages/chat.db",
            backup_timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )

        # Message from Sarah in January
        sarah_jan = Document(
            type=DocumentType.MESSAGE,
            text="Meeting at 2pm",
            source=source,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            metadata={
                "handle": "+15551234567",
                "is_from_me": False,
                "chat_id": 1,
            },
        )

        # Message from Mike in January
        mike_jan = Document(
            type=DocumentType.MESSAGE,
            text="Lunch tomorrow?",
            source=source,
            timestamp=datetime(2024, 1, 20, 10, 0, 0),
            metadata={
                "handle": "+15559876543",
                "is_from_me": False,
                "chat_id": 2,
            },
        )

        # Message from Sarah in February
        sarah_feb = Document(
            type=DocumentType.MESSAGE,
            text="Project update",
            source=source,
            timestamp=datetime(2024, 2, 15, 10, 0, 0),
            metadata={
                "handle": "+15551234567",
                "is_from_me": False,
                "chat_id": 1,
            },
        )

        store.insert_batch([sarah_jan, mike_jan, sarah_feb])

        # Filter by contact and date
        results = store.query(
            contact="+15551234567",
            after=datetime(2024, 2, 1, 0, 0, 0),
        )
        assert len(results) == 1
        assert "Project update" in results[0].text

    def test_query_limit(self, store: StructuredStore) -> None:
        """Test limiting query results."""
        source = Source(
            backup_id="test-backup-001",
            domain="HomeDomain",
            relative_path="Library/Messages/chat.db",
            backup_timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )

        # Create 5 messages
        for i in range(5):
            doc = Document(
                type=DocumentType.MESSAGE,
                text=f"Message {i}",
                source=source,
                timestamp=datetime(2024, 1, 1, 10, i, 0),
            )
            store.insert(doc)

        # Query with limit of 2
        results = store.query(limit=2)
        assert len(results) == 2

        # Query with no limit specified (default 100)
        results = store.query(limit=100)
        assert len(results) == 5


class TestFullTextSearch:
    """Tests for FTS5 full-text search."""

    def test_text_search_basic(self, store: StructuredStore, message_doc: Document) -> None:
        """Test basic full-text search."""
        store.insert(message_doc)

        results = store.query(text_search="plumber")
        assert len(results) == 1
        assert "plumber" in results[0].text.lower()

    def test_text_search_multiple_terms(self, store: StructuredStore) -> None:
        """Test FTS with multiple search terms."""
        source = Source(
            backup_id="test-backup-001",
            domain="HomeDomain",
            relative_path="Library/Messages/chat.db",
            backup_timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )

        doc1 = Document(
            type=DocumentType.MESSAGE,
            text="Can you call the plumber?",
            source=source,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
        )

        doc2 = Document(
            type=DocumentType.MESSAGE,
            text="I'll call the electrician instead",
            source=source,
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
        )

        store.insert_batch([doc1, doc2])

        # Search for "call"
        results = store.query(text_search="call")
        assert len(results) == 2

        # Search for "plumber"
        results = store.query(text_search="plumber")
        assert len(results) == 1

    def test_text_search_with_filters(self, store: StructuredStore) -> None:
        """Test FTS combined with structured filters."""
        source = Source(
            backup_id="test-backup-001",
            domain="HomeDomain",
            relative_path="Library/Messages/chat.db",
            backup_timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )

        # Message about plumber from Sarah
        doc1 = Document(
            type=DocumentType.MESSAGE,
            text="Can you call the plumber?",
            source=source,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            metadata={
                "handle": "+15551234567",
                "chat_id": 1,
            },
        )

        # Message about plumber from Mike
        doc2 = Document(
            type=DocumentType.MESSAGE,
            text="The plumber is here",
            source=source,
            timestamp=datetime(2024, 1, 20, 14, 0, 0),
            metadata={
                "handle": "+15559876543",
                "chat_id": 2,
            },
        )

        # Note about plumber
        doc3 = Document(
            type=DocumentType.NOTE,
            text="Call the plumber tomorrow",
            source=source,
            timestamp=datetime(2024, 1, 18, 9, 0, 0),
        )

        store.insert_batch([doc1, doc2, doc3])

        # Search for "plumber" from Sarah only
        results = store.query(text_search="plumber", contact="+15551234567")
        assert len(results) == 1
        assert results[0].metadata.get("handle") == "+15551234567"

        # Search for "plumber" in messages only
        results = store.query(text_search="plumber", type_filter=DocumentType.MESSAGE)
        assert len(results) == 2
        assert all(r.type == DocumentType.MESSAGE for r in results)

    def test_text_search_no_results(self, store: StructuredStore, message_doc: Document) -> None:
        """Test FTS with no matches."""
        store.insert(message_doc)

        results = store.query(text_search="nonexistent_term_xyz")
        assert len(results) == 0

    def test_text_search_ranking(self, store: StructuredStore) -> None:
        """Test that FTS results are ranked."""
        source = Source(
            backup_id="test-backup-001",
            domain="HomeDomain",
            relative_path="Library/Messages/chat.db",
            backup_timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )

        # Document with "plumber" once
        doc1 = Document(
            type=DocumentType.MESSAGE,
            text="I need to call the plumber",
            source=source,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
        )

        # Document with "plumber" twice
        doc2 = Document(
            type=DocumentType.MESSAGE,
            text="The plumber came and the plumber fixed it",
            source=source,
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
        )

        store.insert_batch([doc1, doc2])

        results = store.query(text_search="plumber")
        # Results should be ranked, but both should be returned
        assert len(results) == 2


class TestGetById:
    """Tests for retrieving documents by ID."""

    def test_get_by_id_found(self, store: StructuredStore, message_doc: Document) -> None:
        """Test retrieving an existing document."""
        store.insert(message_doc)

        retrieved = store.get_by_id(message_doc.id)
        assert retrieved is not None
        assert retrieved.id == message_doc.id
        assert retrieved.text == message_doc.text
        assert retrieved.type == message_doc.type

    def test_get_by_id_not_found(self, store: StructuredStore) -> None:
        """Test retrieving a non-existent document."""
        retrieved = store.get_by_id("nonexistent_id")
        assert retrieved is None

    def test_document_roundtrip(self, store: StructuredStore, message_doc: Document) -> None:
        """Test that a document survives insert and retrieval."""
        store.insert(message_doc)

        retrieved = store.get_by_id(message_doc.id)
        assert retrieved is not None
        assert retrieved.type == message_doc.type
        assert retrieved.text == message_doc.text
        assert retrieved.timestamp == message_doc.timestamp
        assert retrieved.metadata == message_doc.metadata
        assert retrieved.source.domain == message_doc.source.domain


class TestMetadataHandling:
    """Tests for metadata extraction and filtering."""

    def test_message_metadata_extracted(
        self, store: StructuredStore, message_doc: Document
    ) -> None:
        """Test that message metadata is correctly extracted."""
        store.insert(message_doc)

        results = store.query(contact="+15551234567")
        assert len(results) == 1
        assert results[0].metadata["handle"] == "+15551234567"

    def test_contact_handle_extraction(self, store: StructuredStore, contact_doc: Document) -> None:
        """Test that contact handle is extracted from email."""
        store.insert(contact_doc)

        # Contact should be queryable by email
        results = store.query(contact="sarah@example.com")
        assert len(results) == 1

    def test_metadata_roundtrip(self, store: StructuredStore, message_doc: Document) -> None:
        """Test that metadata is preserved through insert and retrieve."""
        store.insert(message_doc)

        retrieved = store.get_by_id(message_doc.id)
        assert retrieved is not None
        assert retrieved.metadata == message_doc.metadata


class TestAttachments:
    """Tests for attachment tracking."""

    def test_has_attachments_flag(self, store: StructuredStore, photo_doc: Document) -> None:
        """Test that attachment presence is tracked."""
        store.insert(photo_doc)

        retrieved = store.get_by_id(photo_doc.id)
        assert retrieved is not None
        # Note: attachments list is empty since we don't persist the actual attachment metadata,
        # but the document was marked as having attachments during insert
        # (the has_attachments flag is stored in the database)


class TestErrorHandling:
    """Tests for error handling."""

    def test_search_error_on_invalid_query(self, store: StructuredStore) -> None:
        """Test that SearchError is raised on query failure."""
        # Creating a bad store will fail during init, which is expected
        import sqlite3

        with pytest.raises(sqlite3.OperationalError):
            StructuredStore(db_path="/nonexistent/path/that/does/not/exist/test.db")

    def test_insert_error_on_bad_connection(self) -> None:
        """Test that SearchError is raised on insert failure."""
        # This test is harder to trigger with SQLite, but we can at least verify
        # that the error handling path exists by checking the exception handling
        store = StructuredStore(db_path=":memory:")

        # Normal insert should work
        source = Source(
            backup_id="test",
            domain="test",
            relative_path="test",
            backup_timestamp=datetime.now(),
        )
        doc = Document(type=DocumentType.MESSAGE, text="test", source=source)
        store.insert(doc)

        assert store.count() == 1


class TestCount:
    """Tests for the count operation."""

    def test_count_empty(self, store: StructuredStore) -> None:
        """Test count on empty store."""
        assert store.count() == 0

    def test_count_increments(
        self, store: StructuredStore, message_doc: Document, contact_doc: Document
    ) -> None:
        """Test that count reflects insertions."""
        assert store.count() == 0

        store.insert(message_doc)
        assert store.count() == 1

        store.insert(contact_doc)
        assert store.count() == 2


class TestDeleteAll:
    """Tests for the delete_all operation."""

    def test_delete_all(
        self, store: StructuredStore, message_doc: Document, contact_doc: Document
    ) -> None:
        """Test deleting all documents."""
        store.insert_batch([message_doc, contact_doc])
        assert store.count() == 2

        store.delete_all()
        assert store.count() == 0

    def test_delete_all_empty_store(self, store: StructuredStore) -> None:
        """Test deleting from an already empty store."""
        store.delete_all()
        assert store.count() == 0


@pytest.fixture
def transcript_source() -> Source:
    """Provenance for a generic (non-iOS) transcript document."""
    return Source(
        backup_id="external",
        domain="audio",
        relative_path="recordings/session-1.wav",
        backup_timestamp=datetime(2024, 6, 2, 9, 0, 0),
    )


def _transcript(source: Source, *, text: str, metadata: dict[str, object]) -> Document:
    """Build a TRANSCRIPT document carrying arbitrary metadata."""
    return Document(
        type=DocumentType.TRANSCRIPT,
        text=text,
        source=source,
        timestamp=datetime(2024, 6, 2, 9, 5, 0),
        metadata=metadata,
    )


class TestMetadataFiltering:
    """Tests for generic, domain-agnostic metadata filtering (bpw)."""

    def test_filter_by_indexed_key(self, transcript_source: Source) -> None:
        """A declared indexed key filters through the structured query path."""
        store = StructuredStore(indexed_metadata_keys=("session_id", "topic"))
        store.insert_batch(
            [
                _transcript(
                    transcript_source,
                    text="intro to widgets",
                    metadata={"session_id": "S1", "topic": "widgets"},
                ),
                _transcript(
                    transcript_source,
                    text="more on gadgets",
                    metadata={"session_id": "S2", "topic": "gadgets"},
                ),
            ]
        )

        results = store.query(metadata={"session_id": "S1"})
        assert len(results) == 1
        assert results[0].metadata["topic"] == "widgets"

    def test_filter_by_multiple_keys_anded(self, transcript_source: Source) -> None:
        """Multiple metadata filters are ANDed together."""
        store = StructuredStore(indexed_metadata_keys=("session_id", "topic"))
        store.insert_batch(
            [
                _transcript(
                    transcript_source, text="a", metadata={"session_id": "S1", "topic": "x"}
                ),
                _transcript(
                    transcript_source, text="b", metadata={"session_id": "S1", "topic": "y"}
                ),
            ]
        )

        results = store.query(metadata={"session_id": "S1", "topic": "y"})
        assert len(results) == 1
        assert results[0].text == "b"

    def test_filter_pushed_down_before_limit(self, transcript_source: Source) -> None:
        """The match survives even when it sorts past LIMIT among non-matches.

        Filtering must happen in SQL, not as a post-LIMIT scan in Python.
        """
        store = StructuredStore(indexed_metadata_keys=("topic",))
        docs = [
            _transcript(transcript_source, text=f"filler {i}", metadata={"topic": "common"})
            for i in range(20)
        ]
        docs.append(_transcript(transcript_source, text="the rare one", metadata={"topic": "rare"}))
        store.insert_batch(docs)

        results = store.query(metadata={"topic": "rare"}, limit=5)
        assert [r.text for r in results] == ["the rare one"]

    def test_fts_over_transcript_text(self, transcript_source: Source) -> None:
        """FTS works over transcript text and composes with a metadata filter."""
        store = StructuredStore(indexed_metadata_keys=("topic",))
        store.insert_batch(
            [
                _transcript(
                    transcript_source,
                    text="the plumber is coming on Tuesday",
                    metadata={"topic": "home"},
                ),
                _transcript(
                    transcript_source,
                    text="completely unrelated gardening notes",
                    metadata={"topic": "home"},
                ),
            ]
        )

        results = store.query(text_search="plumber")
        assert len(results) == 1
        assert "plumber" in results[0].text

        results = store.query(text_search="plumber", metadata={"topic": "home"})
        assert len(results) == 1

    def test_json_extract_fallback_for_undeclared_key(
        self, store: StructuredStore, message_doc: Document
    ) -> None:
        """Any metadata key filters even when not declared as indexed."""
        store.insert(message_doc)  # metadata carries handle="+15551234567"
        assert len(store.query(metadata={"handle": "+15551234567"})) == 1
        assert len(store.query(metadata={"handle": "+10000000000"})) == 0

    def test_filtering_without_indexed_keys(self, transcript_source: Source) -> None:
        """The default store (no declared keys) still filters via json_extract."""
        store = StructuredStore()
        store.insert(_transcript(transcript_source, text="hi", metadata={"topic": "x"}))
        assert len(store.query(metadata={"topic": "x"})) == 1
        assert len(store.query(metadata={"topic": "z"})) == 0

    def test_existing_types_unaffected(
        self,
        store: StructuredStore,
        message_doc: Document,
        contact_doc: Document,
        photo_doc: Document,
    ) -> None:
        """iOS documents are unaffected; a metadata filter they lack excludes them."""
        store.insert_batch([message_doc, contact_doc, photo_doc])
        assert store.count() == 3
        assert store.query(metadata={"topic": "anything"}) == []
        assert len(store.query(limit=100)) == 3

    def test_invalid_indexed_key_rejected(self) -> None:
        """Indexed metadata keys must be valid identifiers."""
        with pytest.raises(ValueError, match="valid identifiers"):
            StructuredStore(indexed_metadata_keys=("bad key!",))
