"""Tests for CalendarExtractor."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from mudline.extractors.calendar import CalendarExtractor
from mudline.foundation.manifest import ManifestResolver
from mudline.models.document import DocumentType

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def extractor() -> CalendarExtractor:
    """Create a CalendarExtractor instance."""
    return CalendarExtractor()


@pytest.fixture
def temp_backup(tmp_path: Path) -> Path:
    """Create a minimal temporary backup with Calendar.sqlitedb."""
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

    # Create Calendar.sqlitedb in the backup
    calendar_file_id = "ccddee1122334455667788990011aabb"
    calendar_rel_path = "Library/Calendar/Calendar.sqlitedb"
    manifest_conn.execute(
        "INSERT INTO Files (fileID, domain, relativePath, flags) VALUES (?, ?, ?, ?)",
        (calendar_file_id, "HomeDomain", calendar_rel_path, 1),
    )
    manifest_conn.commit()
    manifest_conn.close()

    # Create the actual Calendar.sqlitedb file
    calendar_dir = backup_dir / calendar_file_id[:2]
    calendar_dir.mkdir(exist_ok=True)
    calendar_path = calendar_dir / calendar_file_id

    cal_conn = sqlite3.connect(calendar_path)
    cal_conn.execute(
        """
        CREATE TABLE CalendarItem (
            ROWID INTEGER PRIMARY KEY,
            summary TEXT,
            description TEXT,
            start_date REAL,
            end_date REAL,
            all_day INTEGER,
            calendar_id INTEGER
        )
        """
    )
    cal_conn.execute(
        """
        CREATE TABLE Calendar (
            ROWID INTEGER PRIMARY KEY,
            title TEXT
        )
        """
    )
    cal_conn.execute(
        """
        CREATE TABLE Attendee (
            ROWID INTEGER PRIMARY KEY,
            address TEXT,
            item_id INTEGER
        )
        """
    )
    cal_conn.execute(
        """
        CREATE TABLE Recurrence (
            ROWID INTEGER PRIMARY KEY,
            frequency TEXT,
            item_id INTEGER
        )
        """
    )

    # Insert sample calendars
    cal_conn.execute(
        "INSERT INTO Calendar (ROWID, title) VALUES (?, ?)",
        (1, "Work"),
    )
    cal_conn.execute(
        "INSERT INTO Calendar (ROWID, title) VALUES (?, ?)",
        (2, "Personal"),
    )

    # Insert sample events
    cocoa_epoch_2026_02_15 = 792806400.0  # Feb 15, 2026 in Cocoa epoch
    cal_conn.execute(
        """
        INSERT INTO CalendarItem
        (ROWID, summary, description, start_date, end_date, all_day, calendar_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "Team Meeting",
            "Quarterly planning",
            cocoa_epoch_2026_02_15 + 3600,
            cocoa_epoch_2026_02_15 + 7200,
            0,
            1,
        ),
    )

    cal_conn.execute(
        """
        INSERT INTO CalendarItem
        (ROWID, summary, description, start_date, end_date, all_day, calendar_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            2,
            "Birthday",
            None,
            cocoa_epoch_2026_02_15 + 86400,
            cocoa_epoch_2026_02_15 + 86400 + 3600,
            1,
            2,
        ),
    )

    # Insert attendees
    cal_conn.execute(
        "INSERT INTO Attendee (ROWID, address, item_id) VALUES (?, ?, ?)",
        (1, "john@example.com", 1),
    )
    cal_conn.execute(
        "INSERT INTO Attendee (ROWID, address, item_id) VALUES (?, ?, ?)",
        (2, "jane@example.com", 1),
    )

    # Insert recurrence
    cal_conn.execute(
        "INSERT INTO Recurrence (ROWID, frequency, item_id) VALUES (?, ?, ?)",
        (1, "WEEKLY", 1),
    )

    cal_conn.commit()
    cal_conn.close()

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


class TestCalendarExtractor:
    """Test suite for CalendarExtractor."""

    def test_domain_property(self, extractor: CalendarExtractor) -> None:
        """Test that domain property is correct."""
        assert extractor.domain == "HomeDomain"

    def test_data_type_property(self, extractor: CalendarExtractor) -> None:
        """Test that data_type property is correct."""
        assert extractor.data_type == "calendar"

    def test_can_extract_with_calendar_db(
        self, extractor: CalendarExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that can_extract returns True when Calendar.sqlitedb exists."""
        assert extractor.can_extract(resolver) is True

    def test_can_extract_without_calendar_db(
        self, extractor: CalendarExtractor, tmp_path: Path
    ) -> None:
        """Test that can_extract returns False when Calendar.sqlitedb is missing."""
        # Create a minimal backup without Calendar.sqlitedb
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

    def test_extract_events(self, extractor: CalendarExtractor, resolver: ManifestResolver) -> None:
        """Test extracting calendar events."""
        docs = list(extractor.extract(resolver))

        # Should have 2 events
        assert len(docs) == 2

        # All should be CALENDAR type
        assert all(doc.type == DocumentType.CALENDAR for doc in docs)

    def test_event_summary(self, extractor: CalendarExtractor, resolver: ManifestResolver) -> None:
        """Test that event summary is in text content."""
        docs = list(extractor.extract(resolver))

        texts = [doc.text for doc in docs]
        assert any("Team Meeting" in text for text in texts)
        assert any("Birthday" in text for text in texts)

    def test_event_metadata(self, extractor: CalendarExtractor, resolver: ManifestResolver) -> None:
        """Test that event metadata is correctly extracted."""
        docs = list(extractor.extract(resolver))

        # Find the Team Meeting event
        meeting = next(doc for doc in docs if "Team Meeting" in doc.text)
        assert meeting.metadata["is_all_day"] is False
        assert meeting.metadata["recurrence"] == "WEEKLY"
        assert len(meeting.metadata["attendees"]) == 2
        assert "john@example.com" in meeting.metadata["attendees"]

    def test_all_day_event(self, extractor: CalendarExtractor, resolver: ManifestResolver) -> None:
        """Test that all-day events are marked correctly."""
        docs = list(extractor.extract(resolver))

        # Find the Birthday event
        birthday = next(doc for doc in docs if "Birthday" in doc.text)
        assert birthday.metadata["is_all_day"] is True

    def test_timestamp_conversion(
        self, extractor: CalendarExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that Cocoa epoch seconds are correctly converted."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert isinstance(doc.timestamp, datetime)
            assert doc.timestamp.year == 2026
            assert doc.timestamp.month == 2

    def test_end_time_metadata(
        self, extractor: CalendarExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that end_time is correctly set in metadata."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert doc.metadata["end_time"] is not None
            assert isinstance(doc.metadata["end_time"], str)
            # Verify it's a valid ISO datetime string
            datetime.fromisoformat(doc.metadata["end_time"])

    def test_source_provenance(
        self, extractor: CalendarExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that source provenance is correctly set."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert doc.source.backup_id is not None
            assert doc.source.domain == "HomeDomain"
            assert "Library/Calendar/Calendar.sqlitedb" in doc.source.relative_path
            assert doc.source.backup_timestamp is not None

    def test_document_id_uniqueness(
        self, extractor: CalendarExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that each document gets a unique ID."""
        docs = list(extractor.extract(resolver))

        ids = [doc.id for doc in docs]
        assert len(ids) == len(set(ids))  # All IDs should be unique

    def test_extractor_implements_protocol(self, extractor: CalendarExtractor) -> None:
        """Test that CalendarExtractor implements the Extractor protocol."""
        from mudline.models.extractor import Extractor

        assert isinstance(extractor, Extractor)
