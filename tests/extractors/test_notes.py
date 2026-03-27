"""Tests for NoteExtractor."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from mudline.extractors.notes import NoteExtractor
from mudline.foundation.manifest import ManifestResolver
from mudline.models.document import DocumentType

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def extractor() -> NoteExtractor:
    """Create a NoteExtractor instance."""
    return NoteExtractor()


@pytest.fixture
def temp_backup(tmp_path: Path) -> Path:
    """Create a minimal temporary backup with NoteStore.sqlite."""
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

    # Create NoteStore.sqlite in the backup
    notes_file_id = "bbccddee1122334455667788990011aa"
    notes_rel_path = "Library/Notes/NoteStore.sqlite"
    manifest_conn.execute(
        "INSERT INTO Files (fileID, domain, relativePath, flags) VALUES (?, ?, ?, ?)",
        (notes_file_id, "HomeDomain", notes_rel_path, 1),
    )
    manifest_conn.commit()
    manifest_conn.close()

    # Create the actual NoteStore.sqlite file
    notes_dir = backup_dir / notes_file_id[:2]
    notes_dir.mkdir(exist_ok=True)
    notes_path = notes_dir / notes_file_id

    notes_conn = sqlite3.connect(notes_path)
    notes_conn.execute(
        """
        CREATE TABLE ZICCLOUDSYNCINGOBJECT (
            Z_PK INTEGER PRIMARY KEY,
            ZTITLE TEXT,
            ZSNIPPET TEXT,
            ZMODIFICATIONDATE REAL,
            ZCREATIONDATE REAL,
            ZDATA BLOB,
            ZFOLDER INTEGER,
            ZACCOUNT INTEGER,
            Z_ENT INTEGER
        )
        """
    )

    # Insert sample notes
    cocoa_epoch_2026_02_15 = 792806400.0  # Feb 15, 2026 in Cocoa epoch
    notes_conn.execute(
        """
        INSERT INTO ZICCLOUDSYNCINGOBJECT
        (Z_PK, ZTITLE, ZSNIPPET, ZMODIFICATIONDATE, ZCREATIONDATE, ZDATA, ZFOLDER, Z_ENT)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "Shopping List",
            "Milk\nBread\nEggs",
            cocoa_epoch_2026_02_15,
            cocoa_epoch_2026_02_15,
            b"sample_data",
            None,
            1,
        ),
    )

    notes_conn.execute(
        """
        INSERT INTO ZICCLOUDSYNCINGOBJECT
        (Z_PK, ZTITLE, ZSNIPPET, ZMODIFICATIONDATE, ZCREATIONDATE, ZDATA, ZFOLDER, Z_ENT)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            2,
            "Meeting Notes",
            "Discuss Q1 roadmap\nReview metrics",
            cocoa_epoch_2026_02_15 + 7200,
            cocoa_epoch_2026_02_15 + 3600,
            None,
            1,
            1,
        ),
    )

    # Folder entry (should be skipped by extractor)
    notes_conn.execute(
        """
        INSERT INTO ZICCLOUDSYNCINGOBJECT
        (Z_PK, ZTITLE, ZSNIPPET, ZMODIFICATIONDATE, ZCREATIONDATE, ZDATA, ZFOLDER, Z_ENT)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            3,
            "Folder",
            None,
            cocoa_epoch_2026_02_15,
            cocoa_epoch_2026_02_15,
            None,
            None,
            2,
        ),
    )

    notes_conn.commit()
    notes_conn.close()

    # Create Info.plist
    import plistlib

    info_plist = backup_dir / "Info.plist"
    with open(info_plist, "wb") as f:
        plistlib.dump(
            {"Last Backup Date": datetime(2026, 2, 15, 10, 0, 0)}, f
        )

    return backup_dir


@pytest.fixture
def resolver(temp_backup: Path) -> ManifestResolver:
    """Create a ManifestResolver for the temporary backup."""
    return ManifestResolver(temp_backup)


class TestNoteExtractor:
    """Test suite for NoteExtractor."""

    def test_domain_property(self, extractor: NoteExtractor) -> None:
        """Test that domain property is correct."""
        assert extractor.domain == "HomeDomain"

    def test_data_type_property(self, extractor: NoteExtractor) -> None:
        """Test that data_type property is correct."""
        assert extractor.data_type == "note"

    def test_can_extract_with_notes_db(
        self, extractor: NoteExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that can_extract returns True when NoteStore.sqlite exists."""
        assert extractor.can_extract(resolver) is True

    def test_can_extract_without_notes_db(
        self, extractor: NoteExtractor, tmp_path: Path
    ) -> None:
        """Test that can_extract returns False when NoteStore.sqlite is missing."""
        # Create a minimal backup without NoteStore.sqlite
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

    def test_extract_notes(
        self, extractor: NoteExtractor, resolver: ManifestResolver
    ) -> None:
        """Test extracting notes from the temporary backup."""
        docs = list(extractor.extract(resolver))

        # Should have 2 notes (folder is skipped)
        assert len(docs) == 2

        # All should be NOTE type
        assert all(doc.type == DocumentType.NOTE for doc in docs)

    def test_note_text_content(
        self, extractor: NoteExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that note text is correctly extracted."""
        docs = list(extractor.extract(resolver))

        texts = [doc.text for doc in docs]
        assert "Milk" in texts[0] or "Milk\nBread\nEggs" in texts[0]
        assert "Discuss" in texts[1] or "Review metrics" in texts[1]

    def test_timestamp_conversion(
        self, extractor: NoteExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that Cocoa epoch seconds are correctly converted."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert isinstance(doc.timestamp, datetime)
            assert doc.timestamp.year == 2026
            assert doc.timestamp.month == 2

    def test_has_attachments_metadata(
        self, extractor: NoteExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that has_attachments metadata is correctly set."""
        docs = list(extractor.extract(resolver))

        # First note has ZDATA
        assert docs[0].metadata["has_attachments"] is True
        # Second note has no ZDATA
        assert docs[1].metadata["has_attachments"] is False

    def test_source_provenance(
        self, extractor: NoteExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that source provenance is correctly set."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert doc.source.backup_id is not None
            assert doc.source.domain == "HomeDomain"
            assert "Library/Notes/NoteStore.sqlite" in doc.source.relative_path
            assert doc.source.backup_timestamp is not None

    def test_document_id_uniqueness(
        self, extractor: NoteExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that each document gets a unique ID."""
        docs = list(extractor.extract(resolver))

        ids = [doc.id for doc in docs]
        assert len(ids) == len(set(ids))  # All IDs should be unique

    def test_extractor_implements_protocol(
        self, extractor: NoteExtractor
    ) -> None:
        """Test that NoteExtractor implements the Extractor protocol."""
        from mudline.models.extractor import Extractor

        assert isinstance(extractor, Extractor)
