"""Tests for VoicemailExtractor."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from mudline.extractors.voicemail import VoicemailExtractor
from mudline.foundation.manifest import ManifestResolver
from mudline.models.document import DocumentType

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def extractor() -> VoicemailExtractor:
    """Create a VoicemailExtractor instance."""
    return VoicemailExtractor()


@pytest.fixture
def temp_backup(tmp_path: Path) -> Path:
    """Create a minimal temporary backup with voicemail.db."""
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

    # Create voicemail.db in the backup
    voicemail_file_id = "ee1122334455667788990011aabbccdd"
    voicemail_rel_path = "Library/Voicemail/voicemail.db"
    manifest_conn.execute(
        "INSERT INTO Files (fileID, domain, relativePath, flags) VALUES (?, ?, ?, ?)",
        (voicemail_file_id, "HomeDomain", voicemail_rel_path, 1),
    )
    manifest_conn.commit()
    manifest_conn.close()

    # Create the actual voicemail.db file
    voicemail_dir = backup_dir / voicemail_file_id[:2]
    voicemail_dir.mkdir(exist_ok=True)
    voicemail_path = voicemail_dir / voicemail_file_id

    vm_conn = sqlite3.connect(voicemail_path)
    vm_conn.execute(
        """
        CREATE TABLE voicemail (
            ROWID INTEGER PRIMARY KEY,
            sender TEXT,
            date REAL,
            duration REAL,
            trashed_date REAL,
            transcription TEXT,
            receiver TEXT
        )
        """
    )

    # Insert sample voicemails
    cocoa_epoch_2026_02_15 = 792806400.0  # Feb 15, 2026 in Cocoa epoch
    vm_conn.execute(
        """
        INSERT INTO voicemail
        (ROWID, sender, date, duration, trashed_date, transcription, receiver)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "+15551234567",
            cocoa_epoch_2026_02_15,
            42.5,
            None,
            "Hey, it's John. Call me back when you get a chance.",
            None,
        ),
    )

    vm_conn.execute(
        """
        INSERT INTO voicemail
        (ROWID, sender, date, duration, trashed_date, transcription, receiver)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            2,
            "+15559876543",
            cocoa_epoch_2026_02_15 + 7200,
            28.0,
            None,
            None,
            None,
        ),
    )

    # Insert a deleted voicemail (should be skipped)
    vm_conn.execute(
        """
        INSERT INTO voicemail
        (ROWID, sender, date, duration, trashed_date, transcription, receiver)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            3,
            "+15557654321",
            cocoa_epoch_2026_02_15 - 86400,
            15.0,
            cocoa_epoch_2026_02_15,
            "This message was deleted",
            None,
        ),
    )

    vm_conn.commit()
    vm_conn.close()

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


class TestVoicemailExtractor:
    """Test suite for VoicemailExtractor."""

    def test_domain_property(self, extractor: VoicemailExtractor) -> None:
        """Test that domain property is correct."""
        assert extractor.domain == "HomeDomain"

    def test_data_type_property(self, extractor: VoicemailExtractor) -> None:
        """Test that data_type property is correct."""
        assert extractor.data_type == "voicemail"

    def test_can_extract_with_voicemail_db(
        self, extractor: VoicemailExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that can_extract returns True when voicemail.db exists."""
        assert extractor.can_extract(resolver) is True

    def test_can_extract_without_voicemail_db(
        self, extractor: VoicemailExtractor, tmp_path: Path
    ) -> None:
        """Test that can_extract returns False when voicemail.db is missing."""
        # Create a minimal backup without voicemail.db
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

    def test_extract_voicemails(
        self, extractor: VoicemailExtractor, resolver: ManifestResolver
    ) -> None:
        """Test extracting voicemails from the temporary backup."""
        docs = list(extractor.extract(resolver))

        # Should have 2 voicemails (deleted one should be skipped)
        assert len(docs) == 2

        # All should be VOICEMAIL type
        assert all(doc.type == DocumentType.VOICEMAIL for doc in docs)

    def test_voicemail_with_transcription(
        self, extractor: VoicemailExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that voicemail with transcription uses transcription as text."""
        docs = list(extractor.extract(resolver))

        # Find the voicemail with transcription
        transcribed = next((doc for doc in docs if "Call me back" in doc.text), None)
        assert transcribed is not None
        assert "Hey, it's John" in transcribed.text

    def test_voicemail_without_transcription(
        self, extractor: VoicemailExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that voicemail without transcription shows sender."""
        docs = list(extractor.extract(resolver))

        # Find the voicemail without transcription
        no_trans = next(
            (doc for doc in docs if "+15559876543" in doc.metadata.get("handle", "")),
            None,
        )
        assert no_trans is not None
        assert "Voicemail from" in no_trans.text or "+15559876543" in no_trans.text

    def test_voicemail_metadata(
        self, extractor: VoicemailExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that voicemail metadata is correctly extracted."""
        docs = list(extractor.extract(resolver))

        # Find the first voicemail
        vm1 = next((doc for doc in docs if "John" in doc.text), None)
        assert vm1 is not None
        assert vm1.metadata["handle"] == "+15551234567"
        assert vm1.metadata["duration_seconds"] == 42
        expected_text = "Hey, it's John. Call me back when you get a chance."
        assert vm1.metadata["transcription"] == expected_text

    def test_deleted_voicemails_skipped(
        self, extractor: VoicemailExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that deleted voicemails (with trashed_date) are skipped."""
        docs = list(extractor.extract(resolver))

        # Verify the deleted message is not in the results
        texts = [doc.text for doc in docs]
        assert not any("This message was deleted" in text for text in texts)

    def test_timestamp_conversion(
        self, extractor: VoicemailExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that Cocoa epoch seconds are correctly converted."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert isinstance(doc.timestamp, datetime)
            assert doc.timestamp.year == 2026
            assert doc.timestamp.month == 2

    def test_source_provenance(
        self, extractor: VoicemailExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that source provenance is correctly set."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert doc.source.backup_id is not None
            assert doc.source.domain == "HomeDomain"
            assert "Library/Voicemail/voicemail.db" in doc.source.relative_path
            assert doc.source.backup_timestamp is not None

    def test_document_id_uniqueness(
        self, extractor: VoicemailExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that each document gets a unique ID."""
        docs = list(extractor.extract(resolver))

        ids = [doc.id for doc in docs]
        assert len(ids) == len(set(ids))  # All IDs should be unique

    def test_extractor_implements_protocol(self, extractor: VoicemailExtractor) -> None:
        """Test that VoicemailExtractor implements the Extractor protocol."""
        from mudline.models.extractor import Extractor

        assert isinstance(extractor, Extractor)
