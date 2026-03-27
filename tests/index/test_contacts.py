"""Tests for the contact resolution index."""

from __future__ import annotations

from datetime import datetime

import pytest

from mudline.index.contacts import ContactIndex
from mudline.models.document import Document, DocumentType, Source


@pytest.fixture
def sample_source() -> Source:
    """Create a sample Source for testing."""
    return Source(
        backup_id="test-backup-123",
        domain="HomeDomain",
        relative_path="Library/AddressBook/AddressBook.sqlitedb",
        backup_timestamp=datetime.now(),
    )


@pytest.fixture
def sample_contact_documents(sample_source: Source) -> list[Document]:
    """Create sample contact documents."""
    return [
        Document(
            type=DocumentType.CONTACT,
            text="John Smith — Acme Corp — +15551234567 — john@example.com",
            source=sample_source,
            metadata={
                "phones": ["+15551234567"],
                "emails": ["john@example.com"],
                "handles": ["john.smith"],
            },
        ),
        Document(
            type=DocumentType.CONTACT,
            text="Sarah Johnson — Tech Inc — +15559876543 — sarah@example.com",
            source=sample_source,
            metadata={
                "phones": ["+15559876543"],
                "emails": ["sarah@example.com"],
                "handles": ["sarahj"],
            },
        ),
        Document(
            type=DocumentType.CONTACT,
            text="Bob Lee — +15551111111 — bob@example.com",
            source=sample_source,
            metadata={
                "phones": ["+15551111111"],
                "emails": ["bob@example.com"],
                "handles": [],
            },
        ),
    ]


class TestContactIndex:
    """Test the ContactIndex class."""

    def test_load_from_documents(
        self, sample_contact_documents: list[Document]
    ) -> None:
        """Test loading contacts from documents."""
        index = ContactIndex()
        index.load_from_documents(sample_contact_documents)

        assert index.count() == 3

    def test_lookup_by_phone(self, sample_contact_documents: list[Document]) -> None:
        """Test looking up a contact by phone number."""
        index = ContactIndex()
        index.load_from_documents(sample_contact_documents)

        # Exact match
        name = index.lookup("+15551234567")
        assert name == "John Smith"

        # With formatting (should normalize)
        name = index.lookup("+1 (555) 123-4567")
        assert name == "John Smith"

    def test_lookup_by_email(self, sample_contact_documents: list[Document]) -> None:
        """Test looking up a contact by email."""
        index = ContactIndex()
        index.load_from_documents(sample_contact_documents)

        name = index.lookup("john@example.com")
        assert name == "John Smith"

        # Case insensitive
        name = index.lookup("JOHN@EXAMPLE.COM")
        assert name == "John Smith"

    def test_lookup_nonexistent_returns_none(
        self, sample_contact_documents: list[Document]
    ) -> None:
        """Test that looking up a nonexistent handle returns None."""
        index = ContactIndex()
        index.load_from_documents(sample_contact_documents)

        assert index.lookup("+15559999999") is None
        assert index.lookup("nonexistent@example.com") is None

    def test_resolve_exact_match(self, sample_contact_documents: list[Document]) -> None:
        """Test exact name resolution."""
        index = ContactIndex()
        index.load_from_documents(sample_contact_documents)

        handles = index.resolve("John Smith")
        assert "+15551234567" in handles
        assert "john@example.com" in handles
        assert "john.smith" in handles

    def test_resolve_case_insensitive(
        self, sample_contact_documents: list[Document]
    ) -> None:
        """Test that resolution is case-insensitive."""
        index = ContactIndex()
        index.load_from_documents(sample_contact_documents)

        handles = index.resolve("john smith")
        assert "+15551234567" in handles

        handles = index.resolve("JOHN SMITH")
        assert "+15551234567" in handles

    def test_resolve_first_name_only(
        self, sample_contact_documents: list[Document]
    ) -> None:
        """Test resolving by first name only."""
        index = ContactIndex()
        index.load_from_documents(sample_contact_documents)

        handles = index.resolve("John")
        assert "+15551234567" in handles

    def test_resolve_last_name_only(
        self, sample_contact_documents: list[Document]
    ) -> None:
        """Test resolving by last name only."""
        index = ContactIndex()
        index.load_from_documents(sample_contact_documents)

        handles = index.resolve("Smith")
        assert "+15551234567" in handles

    def test_resolve_prefix_match(self, sample_contact_documents: list[Document]) -> None:
        """Test resolving with name prefix."""
        index = ContactIndex()
        index.load_from_documents(sample_contact_documents)

        handles = index.resolve("Jo")
        assert "+15551234567" in handles

        handles = index.resolve("Smi")
        assert "+15551234567" in handles

    def test_resolve_nonexistent_returns_empty(
        self, sample_contact_documents: list[Document]
    ) -> None:
        """Test that resolving a nonexistent name returns empty list."""
        index = ContactIndex()
        index.load_from_documents(sample_contact_documents)

        assert index.resolve("Nonexistent") == []
        assert index.resolve("") == []

    def test_resolve_multiple_matches(self, sample_source: Source) -> None:
        """Test resolving a name that matches multiple contacts."""
        index = ContactIndex()

        # Create documents with overlapping names
        docs = [
            Document(
                type=DocumentType.CONTACT,
                text="John Smith — +15551234567 — john@example.com",
                source=sample_source,
                metadata={
                    "phones": ["+15551234567"],
                    "emails": ["john@example.com"],
                    "handles": [],
                },
            ),
            Document(
                type=DocumentType.CONTACT,
                text="John Doe — +15559876543 — john.doe@example.com",
                source=sample_source,
                metadata={
                    "phones": ["+15559876543"],
                    "emails": ["john.doe@example.com"],
                    "handles": [],
                },
            ),
        ]

        index.load_from_documents(docs)

        handles = index.resolve("John")
        assert len(handles) >= 2

    def test_normalize_handle_phone(self) -> None:
        """Test phone number normalization."""
        index = ContactIndex()

        assert index.normalize_handle("+1 (555) 123-4567") == "+15551234567"
        assert index.normalize_handle("555-123-4567") == "5551234567"
        assert index.normalize_handle("+15551234567") == "+15551234567"

    def test_normalize_handle_email(self) -> None:
        """Test email normalization."""
        index = ContactIndex()

        assert index.normalize_handle("john@example.com") == "john@example.com"
        assert index.normalize_handle("JOHN@EXAMPLE.COM") == "john@example.com"
        assert index.normalize_handle("John@Example.COM") == "john@example.com"

    def test_normalize_handle_empty(self) -> None:
        """Test normalizing empty handle."""
        index = ContactIndex()

        assert index.normalize_handle("") == ""
        assert index.normalize_handle("   ") == ""

    def test_count(self, sample_contact_documents: list[Document]) -> None:
        """Test getting contact count."""
        index = ContactIndex()
        assert index.count() == 0

        index.load_from_documents(sample_contact_documents)
        assert index.count() == 3

    def test_skip_non_contact_documents(self, sample_source: Source) -> None:
        """Test that non-contact documents are skipped."""
        index = ContactIndex()

        docs = [
            Document(
                type=DocumentType.CONTACT,
                text="John Smith — +15551234567 — john@example.com",
                source=sample_source,
                metadata={
                    "phones": ["+15551234567"],
                    "emails": ["john@example.com"],
                    "handles": [],
                },
            ),
            Document(
                type=DocumentType.MESSAGE,
                text="Hello there",
                source=sample_source,
                timestamp=datetime.now(),
                metadata={"handle": "+15551234567"},
            ),
        ]

        index.load_from_documents(docs)
        assert index.count() == 1  # Only the contact should be loaded
