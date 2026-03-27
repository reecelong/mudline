"""Tests for CallHistoryExtractor."""

from __future__ import annotations

from datetime import datetime

from mudline.extractors.calls import (
    CallHistoryExtractor,
    _cocoa_seconds_to_datetime,
    _format_duration,
)
from mudline.foundation.manifest import ManifestResolver
from mudline.models.document import DocumentType


class TestFormatDuration:
    """Test duration formatting."""

    def test_zero_duration(self) -> None:
        """Test formatting of zero duration."""
        assert _format_duration(0.0) == "0s"

    def test_seconds_only(self) -> None:
        """Test formatting of duration in seconds only."""
        assert _format_duration(42.0) == "42s"
        assert _format_duration(5.9) == "5s"

    def test_minutes_and_seconds(self) -> None:
        """Test formatting of duration in minutes and seconds."""
        assert _format_duration(185.0) == "3m 5s"
        assert _format_duration(60.0) == "1m"
        assert _format_duration(61.0) == "1m 1s"

    def test_minutes_only(self) -> None:
        """Test formatting of duration with only minutes."""
        assert _format_duration(120.0) == "2m"
        assert _format_duration(180.0) == "3m"


class TestCocoaSecondsToDatetime:
    """Test Cocoa epoch conversion."""

    def test_cocoa_epoch_zero(self) -> None:
        """Test conversion of Cocoa epoch zero."""
        result = _cocoa_seconds_to_datetime(0.0)
        assert result == datetime(2001, 1, 1)

    def test_cocoa_epoch_offset(self) -> None:
        """Test conversion with offset from Cocoa epoch."""
        # 1 day after epoch
        result = _cocoa_seconds_to_datetime(86400.0)
        assert result == datetime(2001, 1, 2)

    def test_cocoa_epoch_large_offset(self) -> None:
        """Test conversion with large offset (multiple years)."""
        # 25 years after epoch = 2026-01-01 (approximately)
        seconds = 25 * 365.25 * 86400  # 25 years
        result = _cocoa_seconds_to_datetime(seconds)
        # Check it's in 2026
        assert result.year == 2026


class TestCallHistoryExtractor:
    """Test CallHistoryExtractor."""

    def test_extractor_properties(self) -> None:
        """Test extractor domain and data type properties."""
        extractor = CallHistoryExtractor()
        assert extractor.domain == "HomeDomain"
        assert extractor.data_type == DocumentType.CALL.value

    def test_can_extract_with_backup(self, backup_path) -> None:
        """Test can_extract returns True when call history exists."""
        resolver = ManifestResolver(backup_path)
        extractor = CallHistoryExtractor()
        assert extractor.can_extract(resolver)

    def test_can_extract_without_backup(self, tmp_path) -> None:
        """Test can_extract returns False when call history doesn't exist."""
        # Create a minimal backup without call history
        import sqlite3

        manifest_db = tmp_path / "Manifest.db"
        conn = sqlite3.connect(manifest_db)
        conn.execute("""
            CREATE TABLE Files (
                fileID TEXT PRIMARY KEY,
                domain TEXT,
                relativePath TEXT,
                flags INTEGER,
                file BLOB
            )
        """)
        conn.commit()
        conn.close()

        # Create Info and Status plists
        import plistlib

        info = {"Device Name": "Test", "Unique Identifier": "test123"}
        with open(tmp_path / "Info.plist", "wb") as f:
            plistlib.dump(info, f)

        status = {"BackupState": "new", "Date": datetime.now()}
        with open(tmp_path / "Status.plist", "wb") as f:
            plistlib.dump(status, f)

        resolver = ManifestResolver(tmp_path)
        extractor = CallHistoryExtractor()
        assert not extractor.can_extract(resolver)

    def test_extract_all_calls(self, backup_path) -> None:
        """Test extracting all calls from backup."""
        resolver = ManifestResolver(backup_path)
        extractor = CallHistoryExtractor()

        calls = list(extractor.extract(resolver))

        # The fixture has 3 calls
        assert len(calls) == 3

        # Check that all are CALL documents
        for call in calls:
            assert call.type == DocumentType.CALL

        # Check timestamps are sorted ascending
        timestamps = [call.timestamp for call in calls]
        assert timestamps == sorted(timestamps)

    def test_outgoing_call(self, backup_path) -> None:
        """Test outgoing answered call extraction."""
        resolver = ManifestResolver(backup_path)
        extractor = CallHistoryExtractor()

        calls = list(extractor.extract(resolver))

        # First call in fixture: outgoing answered, 185s, +15559876543
        outgoing_call = calls[0]
        assert outgoing_call.metadata["handle"] == "+15559876543"
        assert outgoing_call.metadata["duration_seconds"] == 185
        assert outgoing_call.metadata["call_type"] == "outgoing"
        assert "3m 5s" in outgoing_call.text
        assert "outgoing" in outgoing_call.text
        assert "+15559876543" in outgoing_call.text

    def test_missed_call(self, backup_path) -> None:
        """Test incoming missed call extraction."""
        resolver = ManifestResolver(backup_path)
        extractor = CallHistoryExtractor()

        calls = list(extractor.extract(resolver))

        # Second call in fixture: incoming missed, 0s, +15551234567
        missed_call = calls[1]
        assert missed_call.metadata["handle"] == "+15551234567"
        assert missed_call.metadata["duration_seconds"] == 0
        assert missed_call.metadata["call_type"] == "missed"
        assert "0s" in missed_call.text
        assert "missed" in missed_call.text
        assert "+15551234567" in missed_call.text

    def test_incoming_call(self, backup_path) -> None:
        """Test incoming answered call extraction."""
        resolver = ManifestResolver(backup_path)
        extractor = CallHistoryExtractor()

        calls = list(extractor.extract(resolver))

        # Third call in fixture: incoming answered, 42s, +15551112222
        incoming_call = calls[2]
        assert incoming_call.metadata["handle"] == "+15551112222"
        assert incoming_call.metadata["duration_seconds"] == 42
        assert incoming_call.metadata["call_type"] == "incoming"
        assert "42s" in incoming_call.text
        assert "incoming" in incoming_call.text
        assert "+15551112222" in incoming_call.text

    def test_call_source_provenance(self, backup_path) -> None:
        """Test that call documents have proper source provenance."""
        resolver = ManifestResolver(backup_path)
        extractor = CallHistoryExtractor()

        calls = list(extractor.extract(resolver))
        assert len(calls) > 0

        call = calls[0]
        assert call.source.domain == "HomeDomain"
        assert "CallHistoryDB/CallHistory.storedata/call/" in call.source.relative_path
        assert call.source.backup_id is not None
        assert call.source.backup_timestamp is not None

    def test_call_timestamp_validity(self, backup_path) -> None:
        """Test that call timestamps are valid datetime objects."""
        resolver = ManifestResolver(backup_path)
        extractor = CallHistoryExtractor()

        calls = list(extractor.extract(resolver))

        for call in calls:
            assert isinstance(call.timestamp, datetime)
            # All calls should be in March 2026 based on fixture
            assert call.timestamp.year == 2026
            assert call.timestamp.month == 3

    def test_call_document_deterministic_id(self, backup_path) -> None:
        """Test that call documents have deterministic IDs."""
        resolver = ManifestResolver(backup_path)
        extractor = CallHistoryExtractor()

        calls1 = list(extractor.extract(resolver))
        calls2 = list(extractor.extract(resolver))

        # Same backup, same calls → same IDs
        for call1, call2 in zip(calls1, calls2, strict=True):
            assert call1.id == call2.id

    def test_call_text_content(self, backup_path) -> None:
        """Test that call text content is properly formatted."""
        resolver = ManifestResolver(backup_path)
        extractor = CallHistoryExtractor()

        calls = list(extractor.extract(resolver))

        for call in calls:
            # Every call text should follow the format:
            # "Call with <handle>, <duration>, <call_type>"
            assert call.text.startswith("Call with ")
            assert ", " in call.text
            assert call.metadata["call_type"] in call.text
