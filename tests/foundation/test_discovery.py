"""Unit tests for backup discovery module."""

from __future__ import annotations

import plistlib
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from mudline.exceptions import BackupNotFoundError
from mudline.foundation.discovery import BackupDiscovery, BackupInfo


class TestBackupDiscovery:
    """Tests for BackupDiscovery class."""

    def test_discover_synthetic_backup(self, backup_path: Path) -> None:
        """Discover the synthetic test backup in the fixture directory."""
        discoverer = BackupDiscovery()
        # backup_path IS the backup itself, so discover its parent
        backups = discoverer.discover(backup_path.parent)

        assert len(backups) == 1
        backup = backups[0]

        assert backup.device_name == "Test iPhone"
        assert backup.ios_version == "18.3"
        assert backup.udid == "abcdef1234567890abcdef1234567890abcdef12"
        assert backup.backup_date == datetime(2026, 3, 15, 10, 30, 0)
        assert backup.is_encrypted is False
        assert backup.path == backup_path

    def test_validate_backup_valid(self, backup_path: Path) -> None:
        """Validate a known good backup."""
        discoverer = BackupDiscovery()
        backup = discoverer.validate_backup(backup_path)

        assert isinstance(backup, BackupInfo)
        assert backup.device_name == "Test iPhone"
        assert backup.udid == "abcdef1234567890abcdef1234567890abcdef12"

    def test_validate_backup_not_a_directory(self) -> None:
        """Raise error when path is not a directory."""
        with tempfile.NamedTemporaryFile() as f:
            discoverer = BackupDiscovery()
            with pytest.raises(
                BackupNotFoundError,
                match="Not a directory",
            ):
                discoverer.validate_backup(Path(f.name))

    def test_validate_backup_missing_info_plist(self) -> None:
        """Raise error when Info.plist is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir) / "fake_backup"
            backup_dir.mkdir()

            discoverer = BackupDiscovery()
            with pytest.raises(
                BackupNotFoundError,
                match="No valid iOS backup found",
            ):
                discoverer.validate_backup(backup_dir)

    def test_discover_empty_directory(self) -> None:
        """Discover no backups in an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            discoverer = BackupDiscovery()
            backups = discoverer.discover(Path(tmpdir))

            assert len(backups) == 0

    def test_discover_multiple_backups(self) -> None:
        """Discover multiple backups in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)

            # Create two backup subdirectories
            backup1_dir = parent_dir / "backup1"
            backup2_dir = parent_dir / "backup2"
            backup1_dir.mkdir()
            backup2_dir.mkdir()

            # Add Info.plist to first backup
            info1 = {
                "Device Name": "iPhone 1",
                "Product Version": "18.3",
                "Unique Identifier": "uuid1" + "0" * 36,
                "Last Backup Date": datetime(2026, 3, 15, 10, 30, 0),
            }
            with open(backup1_dir / "Info.plist", "wb") as f:
                plistlib.dump(info1, f)

            # Add Info.plist to second backup
            info2 = {
                "Device Name": "iPhone 2",
                "Product Version": "17.5",
                "Unique Identifier": "uuid2" + "0" * 36,
                "Last Backup Date": datetime(2026, 3, 14, 9, 0, 0),
            }
            with open(backup2_dir / "Info.plist", "wb") as f:
                plistlib.dump(info2, f)

            discoverer = BackupDiscovery()
            backups = discoverer.discover(parent_dir)

            assert len(backups) == 2
            device_names = {b.device_name for b in backups}
            assert device_names == {"iPhone 1", "iPhone 2"}

    def test_discover_mixed_content(self) -> None:
        """Handle directory with backups and non-backup items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)

            # Create a backup subdirectory
            backup_dir = parent_dir / "valid_backup"
            backup_dir.mkdir()
            info = {
                "Device Name": "Valid iPhone",
                "Product Version": "18.3",
                "Unique Identifier": "valid" + "0" * 35,
                "Last Backup Date": datetime(2026, 3, 15, 10, 30, 0),
            }
            with open(backup_dir / "Info.plist", "wb") as f:
                plistlib.dump(info, f)

            # Create a regular file (not a directory)
            (parent_dir / "some_file.txt").touch()

            # Create a directory without Info.plist
            (parent_dir / "not_a_backup").mkdir()

            discoverer = BackupDiscovery()
            backups = discoverer.discover(parent_dir)

            assert len(backups) == 1
            assert backups[0].device_name == "Valid iPhone"

    def test_backup_info_encrypted(self) -> None:
        """Parse encrypted backup correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            backup_dir = parent_dir / "encrypted_backup"
            backup_dir.mkdir()

            info = {
                "Device Name": "Encrypted iPhone",
                "Product Version": "18.3",
                "Unique Identifier": "encrypted" + "0" * 31,
                "Last Backup Date": datetime(2026, 3, 15, 10, 30, 0),
            }
            with open(backup_dir / "Info.plist", "wb") as f:
                plistlib.dump(info, f)

            manifest = {
                "IsEncrypted": True,
                "Version": "10.0",
            }
            with open(backup_dir / "Manifest.plist", "wb") as f:
                plistlib.dump(manifest, f)

            discoverer = BackupDiscovery()
            backups = discoverer.discover(parent_dir)

            assert len(backups) == 1
            assert backups[0].is_encrypted is True

    def test_backup_info_missing_encryption_field(self) -> None:
        """Handle missing IsEncrypted field in Manifest.plist gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            backup_dir = parent_dir / "plain_backup"
            backup_dir.mkdir()

            info = {
                "Device Name": "Plain iPhone",
                "Product Version": "18.3",
                "Unique Identifier": "plain" + "0" * 35,
                "Last Backup Date": datetime(2026, 3, 15, 10, 30, 0),
            }
            with open(backup_dir / "Info.plist", "wb") as f:
                plistlib.dump(info, f)

            manifest = {
                "Version": "10.0",
                # No IsEncrypted field
            }
            with open(backup_dir / "Manifest.plist", "wb") as f:
                plistlib.dump(manifest, f)

            discoverer = BackupDiscovery()
            backups = discoverer.discover(parent_dir)

            assert len(backups) == 1
            assert backups[0].is_encrypted is False

    def test_backup_info_no_manifest_plist(self) -> None:
        """Handle backup without Manifest.plist gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            backup_dir = parent_dir / "no_manifest_backup"
            backup_dir.mkdir()

            info = {
                "Device Name": "No Manifest iPhone",
                "Product Version": "18.3",
                "Unique Identifier": "nomanifest" + "0" * 30,
                "Last Backup Date": datetime(2026, 3, 15, 10, 30, 0),
            }
            with open(backup_dir / "Info.plist", "wb") as f:
                plistlib.dump(info, f)
            # No Manifest.plist

            discoverer = BackupDiscovery()
            backups = discoverer.discover(parent_dir)

            assert len(backups) == 1
            assert backups[0].is_encrypted is False

    def test_backup_info_missing_device_name(self) -> None:
        """Handle missing device name gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            backup_dir = parent_dir / "noname_backup"
            backup_dir.mkdir()

            info = {
                # No "Device Name"
                "Product Version": "18.3",
                "Unique Identifier": "noname" + "0" * 34,
                "Last Backup Date": datetime(2026, 3, 15, 10, 30, 0),
            }
            with open(backup_dir / "Info.plist", "wb") as f:
                plistlib.dump(info, f)

            discoverer = BackupDiscovery()
            backups = discoverer.discover(parent_dir)

            assert len(backups) == 1
            assert backups[0].device_name == "Unknown Device"

    def test_backup_info_missing_udid(self) -> None:
        """Reject backup with missing UDID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir)

            info = {
                "Device Name": "No UDID iPhone",
                "Product Version": "18.3",
                # Missing "Unique Identifier"
                "Last Backup Date": datetime(2026, 3, 15, 10, 30, 0),
            }
            with open(backup_dir / "Info.plist", "wb") as f:
                plistlib.dump(info, f)

            discoverer = BackupDiscovery()
            backups = discoverer.discover(backup_dir.parent)

            # Should not discover a backup without UDID
            assert len(backups) == 0

    def test_backup_info_missing_backup_date(self) -> None:
        """Reject backup with missing backup date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir)

            info = {
                "Device Name": "No Date iPhone",
                "Product Version": "18.3",
                "Unique Identifier": "nodate" + "0" * 34,
                # Missing "Last Backup Date"
            }
            with open(backup_dir / "Info.plist", "wb") as f:
                plistlib.dump(info, f)

            discoverer = BackupDiscovery()
            backups = discoverer.discover(backup_dir.parent)

            # Should not discover a backup without backup date
            assert len(backups) == 0

    def test_backup_info_missing_ios_version(self) -> None:
        """Handle missing iOS version gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            backup_dir = parent_dir / "noversion_backup"
            backup_dir.mkdir()

            info = {
                "Device Name": "No Version iPhone",
                # Missing "Product Version"
                "Unique Identifier": "noversion" + "0" * 31,
                "Last Backup Date": datetime(2026, 3, 15, 10, 30, 0),
            }
            with open(backup_dir / "Info.plist", "wb") as f:
                plistlib.dump(info, f)

            discoverer = BackupDiscovery()
            backups = discoverer.discover(parent_dir)

            assert len(backups) == 1
            assert backups[0].ios_version == "Unknown"

    def test_backup_info_corrupt_plist(self) -> None:
        """Handle corrupt plist files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir)

            # Write invalid plist
            (backup_dir / "Info.plist").write_text("not a valid plist")

            discoverer = BackupDiscovery()
            backups = discoverer.discover(backup_dir.parent)

            # Should not discover backup with corrupt plist
            assert len(backups) == 0

    def test_backup_info_path_stored_correctly(self, backup_path: Path) -> None:
        """Verify backup path is stored correctly in BackupInfo."""
        discoverer = BackupDiscovery()
        backups = discoverer.discover(backup_path.parent)

        assert len(backups) == 1
        assert backups[0].path == backup_path
        assert backups[0].path.is_dir()

    def test_verbose_logging(self, backup_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Verify verbose logging outputs discovery information."""
        discoverer = BackupDiscovery(verbose=True)

        with caplog.at_level("INFO"):
            backups = discoverer.discover(backup_path.parent)

        assert len(backups) == 1
        # Check that discovery info was logged
        assert "Discovered backup" in caplog.text
        assert "Test iPhone" in caplog.text
        assert "abcdef1234567890abcdef1234567890abcdef12" in caplog.text
        assert "18.3" in caplog.text

    def test_backup_info_dataclass_frozen(self, backup_path: Path) -> None:
        """Verify BackupInfo is immutable (frozen dataclass)."""
        discoverer = BackupDiscovery()
        backup = discoverer.validate_backup(backup_path)

        # Attempt to modify should raise an error
        with pytest.raises(AttributeError):
            backup.device_name = "Modified"  # type: ignore

    def test_discover_without_argument_nonexistent_paths(self) -> None:
        """Discover returns empty list when default paths don't exist."""
        # On this test system, default iOS backup locations likely don't exist
        discoverer = BackupDiscovery()
        backups = discoverer.discover(None)

        # Should return list (possibly empty) without raising an error
        assert isinstance(backups, list)

    def test_backup_info_equality(self, backup_path: Path) -> None:
        """Verify BackupInfo can be compared for equality."""
        discoverer = BackupDiscovery()
        backup1 = discoverer.validate_backup(backup_path)
        backup2 = discoverer.validate_backup(backup_path)

        assert backup1 == backup2
        assert backup1.device_name == backup2.device_name
        assert backup1.udid == backup2.udid
