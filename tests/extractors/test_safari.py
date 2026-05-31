"""Tests for SafariExtractor."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from mudline.extractors.safari import SafariExtractor
from mudline.foundation.manifest import ManifestResolver
from mudline.models.document import DocumentType

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def extractor() -> SafariExtractor:
    """Create a SafariExtractor instance."""
    return SafariExtractor()


@pytest.fixture
def temp_backup(tmp_path: Path) -> Path:
    """Create a minimal temporary backup with Safari databases."""
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()

    # Create hash prefix directories
    for i in range(256):
        (backup_dir / f"{i:02x}").mkdir()

    # Create Manifest.db
    manifest_path = backup_dir / "Manifest.db"
    manifest_conn = sqlite3.connect(manifest_path)
    manifest_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Files (
            fileID TEXT PRIMARY KEY,
            domain TEXT,
            relativePath TEXT,
            flags INTEGER
        )
        """
    )

    # Create History.db and Bookmarks.db entries
    history_file_id = "ddee1122334455667788990011aabbcc"
    bookmarks_file_id = "ee1122334455667788990011aabbccdd"

    manifest_conn.execute(
        "INSERT INTO Files (fileID, domain, relativePath, flags) VALUES (?, ?, ?, ?)",
        (history_file_id, "HomeDomain", "Library/Safari/History.db", 1),
    )
    manifest_conn.execute(
        "INSERT INTO Files (fileID, domain, relativePath, flags) VALUES (?, ?, ?, ?)",
        (bookmarks_file_id, "HomeDomain", "Library/Safari/Bookmarks.db", 1),
    )
    manifest_conn.commit()
    manifest_conn.close()

    # Create History.db
    history_dir = backup_dir / history_file_id[:2]
    history_dir.mkdir(exist_ok=True)
    history_path = history_dir / history_file_id

    history_conn = sqlite3.connect(history_path)
    history_conn.execute(
        """
        CREATE TABLE history_items (
            id INTEGER PRIMARY KEY,
            url TEXT,
            visit_count INTEGER
        )
        """
    )
    history_conn.execute(
        """
        CREATE TABLE history_visits (
            id INTEGER PRIMARY KEY,
            history_item INTEGER,
            visit_time REAL,
            title TEXT
        )
        """
    )

    # Insert history items
    history_conn.execute(
        "INSERT INTO history_items (id, url, visit_count) VALUES (?, ?, ?)",
        (1, "https://www.google.com", 42),
    )
    history_conn.execute(
        "INSERT INTO history_items (id, url, visit_count) VALUES (?, ?, ?)",
        (2, "https://github.com", 15),
    )

    # Insert history visits
    cocoa_epoch_2026_02_15 = 792806400.0  # Feb 15, 2026 in Cocoa epoch
    history_conn.execute(
        "INSERT INTO history_visits (id, history_item, visit_time, title) VALUES (?, ?, ?, ?)",
        (1, 1, cocoa_epoch_2026_02_15, "Google"),
    )
    history_conn.execute(
        "INSERT INTO history_visits (id, history_item, visit_time, title) VALUES (?, ?, ?, ?)",
        (2, 2, cocoa_epoch_2026_02_15 + 3600, "GitHub"),
    )

    history_conn.commit()
    history_conn.close()

    # Create Bookmarks.db
    bookmarks_dir = backup_dir / bookmarks_file_id[:2]
    bookmarks_dir.mkdir(exist_ok=True)
    bookmarks_path = bookmarks_dir / bookmarks_file_id

    bookmarks_conn = sqlite3.connect(bookmarks_path)
    bookmarks_conn.execute(
        """
        CREATE TABLE bookmarks (
            id INTEGER PRIMARY KEY,
            title TEXT,
            url TEXT,
            parent INTEGER
        )
        """
    )

    # Insert bookmark folders
    bookmarks_conn.execute(
        "INSERT INTO bookmarks (id, title, url, parent) VALUES (?, ?, ?, ?)",
        (1, "Favorites", None, None),
    )

    # Insert bookmarks
    bookmarks_conn.execute(
        "INSERT INTO bookmarks (id, title, url, parent) VALUES (?, ?, ?, ?)",
        (2, "Stack Overflow", "https://stackoverflow.com", 1),
    )
    bookmarks_conn.execute(
        "INSERT INTO bookmarks (id, title, url, parent) VALUES (?, ?, ?, ?)",
        (3, "MDN", "https://developer.mozilla.org", 1),
    )

    bookmarks_conn.commit()
    bookmarks_conn.close()

    # Create Info.plist
    import plistlib

    info_plist = backup_dir / "Info.plist"
    with open(info_plist, "wb") as f:
        plistlib.dump({"Last Backup Date": datetime(2026, 2, 15, 10, 0, 0)}, f)

    return backup_dir


@pytest.fixture
def resolver(temp_backup: Path) -> ManifestResolver:
    """Create a ManifestResolver for the temporary backup."""
    return ManifestResolver(temp_backup)


class TestSafariExtractor:
    """Test suite for SafariExtractor."""

    def test_domain_property(self, extractor: SafariExtractor) -> None:
        """Test that domain property is correct."""
        assert extractor.domain == "HomeDomain"

    def test_data_type_property(self, extractor: SafariExtractor) -> None:
        """Test that data_type property is correct."""
        assert extractor.data_type == "safari"

    def test_can_extract_with_history_db(
        self, extractor: SafariExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that can_extract returns True when History.db exists."""
        assert extractor.can_extract(resolver) is True

    def test_can_extract_without_history_db(
        self, extractor: SafariExtractor, tmp_path: Path
    ) -> None:
        """Test that can_extract returns False when History.db is missing."""
        # Create a minimal backup without History.db
        backup_dir = tmp_path / "empty_backup"
        backup_dir.mkdir()
        (backup_dir / "00").mkdir()

        manifest_path = backup_dir / "Manifest.db"
        manifest_conn = sqlite3.connect(manifest_path)
        manifest_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS Files (
                fileID TEXT PRIMARY KEY,
                domain TEXT,
                relativePath TEXT,
                flags INTEGER
            )
            """
        )
        manifest_conn.commit()
        manifest_conn.close()

        resolver = ManifestResolver(backup_dir)
        assert extractor.can_extract(resolver) is False

    def test_extract_history_and_bookmarks(
        self, extractor: SafariExtractor, resolver: ManifestResolver
    ) -> None:
        """Test extracting both history and bookmarks."""
        docs = list(extractor.extract(resolver))

        # Should have 2 history items + 2 bookmarks = 4 documents
        assert len(docs) == 4

        # All should be SAFARI type
        assert all(doc.type == DocumentType.SAFARI for doc in docs)

    def test_history_entry_content(
        self, extractor: SafariExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that history entry content is correctly extracted."""
        docs = list(extractor.extract(resolver))

        # Filter to history items only (those with timestamps)
        history_docs = [doc for doc in docs if doc.timestamp is not None]

        texts = [doc.text for doc in history_docs]
        assert any("Google" in text for text in texts)
        assert any("GitHub" in text for text in texts)

    def test_history_metadata(self, extractor: SafariExtractor, resolver: ManifestResolver) -> None:
        """Test that history metadata is correctly extracted."""
        docs = list(extractor.extract(resolver))

        # Find Google history
        google = next(
            (doc for doc in docs if "google.com" in doc.metadata.get("url", "")),
            None,
        )
        assert google is not None
        assert google.metadata["visit_count"] == 42
        assert google.metadata["url"] == "https://www.google.com"

    def test_bookmark_content(self, extractor: SafariExtractor, resolver: ManifestResolver) -> None:
        """Test that bookmark content is correctly extracted."""
        docs = list(extractor.extract(resolver))

        # Filter to bookmarks (those without timestamps)
        bookmark_docs = [doc for doc in docs if doc.timestamp is None]

        texts = [doc.text for doc in bookmark_docs]
        assert any("Stack Overflow" in text for text in texts)
        assert any("MDN" in text for text in texts)

    def test_bookmark_metadata(
        self, extractor: SafariExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that bookmark metadata is correctly extracted."""
        docs = list(extractor.extract(resolver))

        # Find Stack Overflow bookmark
        so = next(
            (doc for doc in docs if "stackoverflow.com" in doc.metadata.get("url", "")),
            None,
        )
        assert so is not None
        assert so.metadata["folder"] == "Favorites"
        assert so.metadata["url"] == "https://stackoverflow.com"

    def test_timestamp_conversion(
        self, extractor: SafariExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that Cocoa epoch seconds are correctly converted."""
        docs = list(extractor.extract(resolver))

        # Check history timestamps
        history_docs = [doc for doc in docs if doc.timestamp is not None]
        for doc in history_docs:
            assert isinstance(doc.timestamp, datetime)
            assert doc.timestamp.year == 2026
            assert doc.timestamp.month == 2

    def test_source_provenance(
        self, extractor: SafariExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that source provenance is correctly set."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert doc.source.backup_id is not None
            assert doc.source.domain == "HomeDomain"
            assert doc.source.backup_timestamp is not None

    def test_document_id_uniqueness(
        self, extractor: SafariExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that each document gets a unique ID."""
        docs = list(extractor.extract(resolver))

        ids = [doc.id for doc in docs]
        assert len(ids) == len(set(ids))  # All IDs should be unique

    def test_extractor_implements_protocol(self, extractor: SafariExtractor) -> None:
        """Test that SafariExtractor implements the Extractor protocol."""
        from mudline.models.extractor import Extractor

        assert isinstance(extractor, Extractor)
