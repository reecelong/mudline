"""Tests for ContactExtractor."""

from __future__ import annotations

import pytest

from mudline.extractors.contacts import ContactExtractor
from mudline.foundation.manifest import ManifestResolver
from mudline.models.document import DocumentType


class TestContactExtractor:
    """Test suite for ContactExtractor."""

    @pytest.fixture
    def extractor(self) -> ContactExtractor:
        """Create a ContactExtractor instance."""
        return ContactExtractor()

    def test_domain_property(self, extractor: ContactExtractor) -> None:
        """Test that domain property returns HomeDomain."""
        assert extractor.domain == "HomeDomain"

    def test_data_type_property(self, extractor: ContactExtractor) -> None:
        """Test that data_type property returns contact."""
        assert extractor.data_type == "contact"

    def test_can_extract_with_valid_backup(self, extractor: ContactExtractor, backup_path) -> None:
        """Test can_extract returns True when AddressBook exists."""
        resolver = ManifestResolver(backup_path)
        assert extractor.can_extract(resolver) is True

    def test_extract_returns_documents(self, extractor: ContactExtractor, backup_path) -> None:
        """Test that extract yields Document objects."""
        resolver = ManifestResolver(backup_path)
        docs = list(extractor.extract(resolver))

        assert len(docs) == 3, "Expected 3 contacts from fixture"
        assert all(doc.type == DocumentType.CONTACT for doc in docs)

    def test_extract_sarah_johnson(self, extractor: ContactExtractor, backup_path) -> None:
        """Test extraction of Sarah Johnson contact."""
        resolver = ManifestResolver(backup_path)
        docs = {doc.text.split(" — ")[0]: doc for doc in extractor.extract(resolver)}

        assert "Sarah Johnson" in docs
        sarah = docs["Sarah Johnson"]

        # Check metadata
        assert sarah.metadata["organization"] == "Acme Corp"
        assert "+15559876543" in sarah.metadata["phones"]
        assert "sarah@example.com" in sarah.metadata["emails"]
        assert "+15559876543" in sarah.metadata["handles"]
        assert "sarah@example.com" in sarah.metadata["handles"]

        # Check text content includes all parts
        assert "Sarah Johnson" in sarah.text
        assert "Acme Corp" in sarah.text
        assert "+15559876543" in sarah.text
        assert "sarah@example.com" in sarah.text

    def test_extract_john_smith(self, extractor: ContactExtractor, backup_path) -> None:
        """Test extraction of John Smith contact."""
        resolver = ManifestResolver(backup_path)
        docs = {doc.text.split(" — ")[0]: doc for doc in extractor.extract(resolver)}

        assert "John Smith" in docs
        john = docs["John Smith"]

        # Check metadata
        assert john.metadata["organization"] is None
        assert "+15551112222" in john.metadata["phones"]
        assert len(john.metadata["emails"]) == 0
        assert "+15551112222" in john.metadata["handles"]

        # Check text doesn't include empty organization or email
        assert "John Smith" in john.text
        assert "+15551112222" in john.text

    def test_extract_mom(self, extractor: ContactExtractor, backup_path) -> None:
        """Test extraction of Mom contact (no last name, no org)."""
        resolver = ManifestResolver(backup_path)
        docs = {doc.text.split(" — ")[0]: doc for doc in extractor.extract(resolver)}

        assert "Mom" in docs
        mom = docs["Mom"]

        # Check metadata
        assert mom.metadata["organization"] is None
        assert "+15551234567" in mom.metadata["phones"]
        assert len(mom.metadata["emails"]) == 0
        assert "+15551234567" in mom.metadata["handles"]

        # Check text
        assert "Mom" in mom.text
        assert "+15551234567" in mom.text

    def test_extract_no_timestamp(self, extractor: ContactExtractor, backup_path) -> None:
        """Test that contacts have no timestamp."""
        resolver = ManifestResolver(backup_path)
        docs = list(extractor.extract(resolver))

        assert all(doc.timestamp is None for doc in docs)

    def test_extract_source_provenance(self, extractor: ContactExtractor, backup_path) -> None:
        """Test that documents include proper source provenance."""
        resolver = ManifestResolver(backup_path)
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert doc.source.domain == "HomeDomain"
            assert doc.source.relative_path == "Library/AddressBook/AddressBook.sqlitedb"
            assert doc.source.backup_timestamp is not None

    def test_get_handle_map_complete(self, extractor: ContactExtractor, backup_path) -> None:
        """Test that get_handle_map returns all handles."""
        resolver = ManifestResolver(backup_path)
        handle_map = extractor.get_handle_map(resolver)

        # Check expected handles
        assert handle_map["+15559876543"] == "Sarah Johnson"
        assert handle_map["sarah@example.com"] == "Sarah Johnson"
        assert handle_map["+15551112222"] == "John Smith"
        assert handle_map["+15551234567"] == "Mom"

        # Should have exactly 4 handles (2 for Sarah, 1 for John, 1 for Mom)
        assert len(handle_map) == 4

    def test_get_handle_map_phone_to_name(self, extractor: ContactExtractor, backup_path) -> None:
        """Test that phone handles map to correct contact name."""
        resolver = ManifestResolver(backup_path)
        handle_map = extractor.get_handle_map(resolver)

        assert handle_map["+15559876543"] == "Sarah Johnson"
        assert handle_map["+15551112222"] == "John Smith"
        assert handle_map["+15551234567"] == "Mom"

    def test_get_handle_map_email_to_name(self, extractor: ContactExtractor, backup_path) -> None:
        """Test that email handles map to correct contact name."""
        resolver = ManifestResolver(backup_path)
        handle_map = extractor.get_handle_map(resolver)

        assert handle_map["sarah@example.com"] == "Sarah Johnson"

    def test_extract_document_has_id(self, extractor: ContactExtractor, backup_path) -> None:
        """Test that extracted documents have deterministic IDs."""
        resolver = ManifestResolver(backup_path)
        docs = list(extractor.extract(resolver))

        # All documents should have non-empty IDs
        assert all(doc.id for doc in docs)

        # Extract again and verify IDs are the same (deterministic)
        docs2 = list(extractor.extract(resolver))
        ids1 = {doc.text.split(" — ")[0]: doc.id for doc in docs}
        ids2 = {doc.text.split(" — ")[0]: doc.id for doc in docs2}

        assert ids1 == ids2

    def test_extract_document_attachments_empty(
        self, extractor: ContactExtractor, backup_path
    ) -> None:
        """Test that contact documents have no attachments."""
        resolver = ManifestResolver(backup_path)
        docs = list(extractor.extract(resolver))

        assert all(len(doc.attachments) == 0 for doc in docs)

    def test_extract_handles_list_not_empty(self, extractor: ContactExtractor, backup_path) -> None:
        """Test that metadata handles list is populated for all contacts."""
        resolver = ManifestResolver(backup_path)
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert len(doc.metadata["handles"]) > 0
            # handles should be a list combining phones and emails
            assert isinstance(doc.metadata["handles"], list)

    def test_extract_with_missing_backup_raises(
        self, extractor: ContactExtractor, tmp_path
    ) -> None:
        """Test that extract raises FileNotFoundError for missing backup."""
        # Create a minimal backup without AddressBook
        manifest_db = tmp_path / "Manifest.db"
        import sqlite3

        conn = sqlite3.connect(manifest_db)
        conn.execute(
            """
            CREATE TABLE Files (
                fileID TEXT PRIMARY KEY,
                domain TEXT,
                relativePath TEXT,
                flags INTEGER,
                file BLOB
            )
            """
        )
        conn.commit()
        conn.close()

        resolver = ManifestResolver(tmp_path)

        with pytest.raises(FileNotFoundError):
            list(extractor.extract(resolver))

    def test_can_extract_with_missing_file_returns_false(
        self, extractor: ContactExtractor, tmp_path
    ) -> None:
        """Test that can_extract returns False if AddressBook doesn't exist."""
        # Create a minimal backup without AddressBook
        manifest_db = tmp_path / "Manifest.db"
        import sqlite3

        conn = sqlite3.connect(manifest_db)
        conn.execute(
            """
            CREATE TABLE Files (
                fileID TEXT PRIMARY KEY,
                domain TEXT,
                relativePath TEXT,
                flags INTEGER,
                file BLOB
            )
            """
        )
        conn.commit()
        conn.close()

        resolver = ManifestResolver(tmp_path)
        assert extractor.can_extract(resolver) is False

    def test_extract_multiple_calls_consistent(
        self, extractor: ContactExtractor, backup_path
    ) -> None:
        """Test that multiple extractions return consistent results."""
        resolver = ManifestResolver(backup_path)
        docs1 = sorted(list(extractor.extract(resolver)), key=lambda d: d.text)
        docs2 = sorted(list(extractor.extract(resolver)), key=lambda d: d.text)

        assert len(docs1) == len(docs2)
        for d1, d2 in zip(docs1, docs2, strict=True):
            assert d1.text == d2.text
            assert d1.metadata == d2.metadata
            assert d1.id == d2.id
