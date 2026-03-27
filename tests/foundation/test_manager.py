"""Tests for BackupManager."""

from __future__ import annotations

import plistlib
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from mudline.foundation.discovery import BackupDiscovery
from mudline.foundation.manager import BackupDiff, BackupManager


def create_synthetic_backup(
    backup_dir: Path,
    device_name: str,
    ios_version: str,
    udid: str,
    backup_date: datetime,
    files: dict[tuple[str, str], str] | None = None,
) -> None:
    """Create a minimal synthetic iOS backup with Info.plist and Manifest.db.

    Args:
        backup_dir: Directory to create the backup in.
        device_name: Device name for Info.plist.
        ios_version: iOS version for Info.plist.
        udid: Device UDID for Info.plist.
        backup_date: Backup date for Info.plist.
        files: Optional dict of (domain, relative_path) -> file_id mappings.
               If None, creates three default files.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Create Info.plist
    info_plist = {
        "Device Name": device_name,
        "Product Version": ios_version,
        "Unique Identifier": udid,
        "Last Backup Date": backup_date,
    }
    with open(backup_dir / "Info.plist", "wb") as f:
        plistlib.dump(info_plist, f)

    # Create Manifest.plist
    manifest_plist = {
        "IsEncrypted": False,
        "Version": "10.0",
    }
    with open(backup_dir / "Manifest.plist", "wb") as f:
        plistlib.dump(manifest_plist, f)

    # Use default files if not provided
    if files is None:
        files = {
            ("HomeDomain", "Library/SMS/sms.db"): ("aabbcc1111111111111111111111111111111111"),
            ("HomeDomain", "Library/AddressBook/AddressBook.sqlitedb"): (
                "ddeeff2222222222222222222222222222222222"
            ),
            ("HomeDomain", "Library/CallHistoryDB/CallHistory.storedata"): (
                "112233333333333333333333333333333333333"
            ),
        }

    # Create Manifest.db with Files table
    db_path = backup_dir / "Manifest.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE Files (fileID TEXT, domain TEXT, relativePath TEXT, flags INTEGER)")

    for (domain, rel_path), file_id in files.items():
        conn.execute(
            "INSERT INTO Files VALUES (?, ?, ?, ?)",
            (file_id, domain, rel_path, 1),  # flags=1 for regular file
        )

    conn.commit()
    conn.close()

    # Create the actual files in 2-char prefix directories
    for file_id in set(fid for fid in files.values()):
        prefix_dir = backup_dir / file_id[:2]
        prefix_dir.mkdir(exist_ok=True)
        file_path = prefix_dir / file_id
        file_path.write_text(f"mock backup data for {file_id}")


class TestBackupManager:
    """Tests for BackupManager class."""

    def test_init_with_default_discovery(self) -> None:
        """Test BackupManager initialization with default discovery."""
        manager = BackupManager()
        assert manager.discovery is not None
        assert isinstance(manager.discovery, BackupDiscovery)

    def test_init_with_custom_discovery(self) -> None:
        """Test BackupManager initialization with custom discovery."""
        custom_discovery = BackupDiscovery(verbose=True)
        manager = BackupManager(discovery=custom_discovery)
        assert manager.discovery is custom_discovery

    def test_scan_empty_directory(self) -> None:
        """Test scan on an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BackupManager()
            manager.scan(Path(tmpdir))

            assert manager.list_devices() == []

    def test_scan_single_backup(self) -> None:
        """Test scan with a single backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            backup_dir = parent_dir / "backup1"

            create_synthetic_backup(
                backup_dir,
                device_name="Test iPhone",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 15, 10, 30, 0),
            )

            manager = BackupManager()
            manager.scan(parent_dir)

            devices = manager.list_devices()
            assert len(devices) == 1
            assert devices[0] == "device001" + "0" * 31

    def test_scan_multiple_backups_same_device(self) -> None:
        """Test scan with multiple backups from the same device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            udid = "device123" + "0" * 31

            # Create three backups with different dates
            backup1_dir = parent_dir / "backup1"
            backup2_dir = parent_dir / "backup2"
            backup3_dir = parent_dir / "backup3"

            create_synthetic_backup(
                backup1_dir,
                device_name="Jane's iPhone",
                ios_version="18.3",
                udid=udid,
                backup_date=datetime(2026, 3, 15, 10, 0, 0),
            )
            create_synthetic_backup(
                backup2_dir,
                device_name="Jane's iPhone",
                ios_version="18.3",
                udid=udid,
                backup_date=datetime(2026, 3, 16, 12, 0, 0),
            )
            create_synthetic_backup(
                backup3_dir,
                device_name="Jane's iPhone",
                ios_version="18.3",
                udid=udid,
                backup_date=datetime(2026, 3, 14, 8, 0, 0),
            )

            manager = BackupManager()
            manager.scan(parent_dir)

            devices = manager.list_devices()
            assert len(devices) == 1
            assert devices[0] == udid

            snapshots = manager.list_snapshots(udid)
            assert len(snapshots) == 3

            # Verify chronological ordering (oldest first)
            assert snapshots[0].backup_date == datetime(2026, 3, 14, 8, 0, 0)
            assert snapshots[1].backup_date == datetime(2026, 3, 15, 10, 0, 0)
            assert snapshots[2].backup_date == datetime(2026, 3, 16, 12, 0, 0)

    def test_scan_multiple_devices(self) -> None:
        """Test scan with backups from multiple devices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)

            udid1 = "device111" + "0" * 31
            udid2 = "device222" + "0" * 31
            udid3 = "device333" + "0" * 31

            create_synthetic_backup(
                parent_dir / "backup1",
                device_name="iPhone 1",
                ios_version="18.3",
                udid=udid1,
                backup_date=datetime(2026, 3, 15, 10, 0, 0),
            )
            create_synthetic_backup(
                parent_dir / "backup2",
                device_name="iPhone 2",
                ios_version="17.5",
                udid=udid2,
                backup_date=datetime(2026, 3, 15, 11, 0, 0),
            )
            create_synthetic_backup(
                parent_dir / "backup3",
                device_name="iPad",
                ios_version="18.2",
                udid=udid3,
                backup_date=datetime(2026, 3, 15, 12, 0, 0),
            )

            manager = BackupManager()
            manager.scan(parent_dir)

            devices = manager.list_devices()
            assert len(devices) == 3
            assert set(devices) == {udid1, udid2, udid3}
            # list_devices should return sorted list
            assert devices == sorted(devices)

    def test_list_snapshots_nonexistent_device(self) -> None:
        """Test list_snapshots for a device that doesn't exist."""
        manager = BackupManager()
        result = manager.list_snapshots("nonexistent_udid")
        assert result == []

    def test_get_latest_nonexistent_device(self) -> None:
        """Test get_latest for a device that doesn't exist."""
        manager = BackupManager()
        result = manager.get_latest("nonexistent_udid")
        assert result is None

    def test_get_latest_single_backup(self) -> None:
        """Test get_latest with a single backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            udid = "device999" + "0" * 31

            backup_dir = parent_dir / "backup1"
            create_synthetic_backup(
                backup_dir,
                device_name="Test iPhone",
                ios_version="18.3",
                udid=udid,
                backup_date=datetime(2026, 3, 15, 10, 0, 0),
            )

            manager = BackupManager()
            manager.scan(parent_dir)

            latest = manager.get_latest(udid)
            assert latest is not None
            assert latest.backup_date == datetime(2026, 3, 15, 10, 0, 0)

    def test_get_latest_multiple_backups(self) -> None:
        """Test get_latest returns the most recent backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            udid = "device888" + "0" * 31

            create_synthetic_backup(
                parent_dir / "backup1",
                device_name="Test",
                ios_version="18.3",
                udid=udid,
                backup_date=datetime(2026, 3, 10, 10, 0, 0),
            )
            create_synthetic_backup(
                parent_dir / "backup2",
                device_name="Test",
                ios_version="18.3",
                udid=udid,
                backup_date=datetime(2026, 3, 20, 10, 0, 0),
            )
            create_synthetic_backup(
                parent_dir / "backup3",
                device_name="Test",
                ios_version="18.3",
                udid=udid,
                backup_date=datetime(2026, 3, 15, 10, 0, 0),
            )

            manager = BackupManager()
            manager.scan(parent_dir)

            latest = manager.get_latest(udid)
            assert latest is not None
            assert latest.backup_date == datetime(2026, 3, 20, 10, 0, 0)

    def test_diff_identical_backups(self) -> None:
        """Test diff between two identical backup file sets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)

            # Create two backups with identical files
            backup1_dir = parent_dir / "backup1"
            backup2_dir = parent_dir / "backup2"

            files = {
                ("HomeDomain", "Library/SMS/sms.db"): ("aabbcc1111111111111111111111111111111111"),
                ("HomeDomain", "Library/Contacts/Contacts.db"): (
                    "ddeeff2222222222222222222222222222222222"
                ),
            }

            create_synthetic_backup(
                backup1_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 15, 10, 0, 0),
                files=files,
            )
            create_synthetic_backup(
                backup2_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 16, 10, 0, 0),
                files=files,
            )

            discovery = BackupDiscovery()
            backup_a = discovery.validate_backup(backup1_dir)
            backup_b = discovery.validate_backup(backup2_dir)

            manager = BackupManager()
            diff = manager.diff(backup_a, backup_b)

            assert diff.added == []
            assert diff.removed == []
            assert diff.changed == []

    def test_diff_files_added(self) -> None:
        """Test diff detects added files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)

            backup1_dir = parent_dir / "backup1"
            backup2_dir = parent_dir / "backup2"

            files_a = {
                ("HomeDomain", "Library/SMS/sms.db"): ("aabbcc1111111111111111111111111111111111"),
            }
            files_b = {
                ("HomeDomain", "Library/SMS/sms.db"): ("aabbcc1111111111111111111111111111111111"),
                ("HomeDomain", "Library/Contacts/Contacts.db"): (
                    "ddeeff2222222222222222222222222222222222"
                ),
                ("CameraRollDomain", "Media/DCIM/photo.jpg"): (
                    "112233333333333333333333333333333333333"
                ),
            }

            create_synthetic_backup(
                backup1_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 15, 10, 0, 0),
                files=files_a,
            )
            create_synthetic_backup(
                backup2_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 16, 10, 0, 0),
                files=files_b,
            )

            discovery = BackupDiscovery()
            backup_a = discovery.validate_backup(backup1_dir)
            backup_b = discovery.validate_backup(backup2_dir)

            manager = BackupManager()
            diff = manager.diff(backup_a, backup_b)

            assert set(diff.added) == {
                "HomeDomain/Library/Contacts/Contacts.db",
                "CameraRollDomain/Media/DCIM/photo.jpg",
            }
            assert diff.removed == []
            assert diff.changed == []

    def test_diff_files_removed(self) -> None:
        """Test diff detects removed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)

            backup1_dir = parent_dir / "backup1"
            backup2_dir = parent_dir / "backup2"

            files_a = {
                ("HomeDomain", "Library/SMS/sms.db"): ("aabbcc1111111111111111111111111111111111"),
                ("HomeDomain", "Library/Contacts/Contacts.db"): (
                    "ddeeff2222222222222222222222222222222222"
                ),
                ("CameraRollDomain", "Media/DCIM/photo.jpg"): (
                    "112233333333333333333333333333333333333"
                ),
            }
            files_b = {
                ("HomeDomain", "Library/SMS/sms.db"): ("aabbcc1111111111111111111111111111111111"),
            }

            create_synthetic_backup(
                backup1_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 15, 10, 0, 0),
                files=files_a,
            )
            create_synthetic_backup(
                backup2_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 16, 10, 0, 0),
                files=files_b,
            )

            discovery = BackupDiscovery()
            backup_a = discovery.validate_backup(backup1_dir)
            backup_b = discovery.validate_backup(backup2_dir)

            manager = BackupManager()
            diff = manager.diff(backup_a, backup_b)

            assert diff.added == []
            assert set(diff.removed) == {
                "HomeDomain/Library/Contacts/Contacts.db",
                "CameraRollDomain/Media/DCIM/photo.jpg",
            }
            assert diff.changed == []

    def test_diff_files_changed(self) -> None:
        """Test diff detects changed files (same path, different file_id)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)

            backup1_dir = parent_dir / "backup1"
            backup2_dir = parent_dir / "backup2"

            files_a = {
                ("HomeDomain", "Library/SMS/sms.db"): ("aabbcc1111111111111111111111111111111111"),
                ("HomeDomain", "Library/Contacts/Contacts.db"): (
                    "ddeeff2222222222222222222222222222222222"
                ),
            }
            files_b = {
                ("HomeDomain", "Library/SMS/sms.db"): ("aabbcc1111111111111111111111111111111111"),
                ("HomeDomain", "Library/Contacts/Contacts.db"): (
                    "aabbcc3333333333333333333333333333333333"  # Changed
                ),
            }

            create_synthetic_backup(
                backup1_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 15, 10, 0, 0),
                files=files_a,
            )
            create_synthetic_backup(
                backup2_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 16, 10, 0, 0),
                files=files_b,
            )

            discovery = BackupDiscovery()
            backup_a = discovery.validate_backup(backup1_dir)
            backup_b = discovery.validate_backup(backup2_dir)

            manager = BackupManager()
            diff = manager.diff(backup_a, backup_b)

            assert diff.added == []
            assert diff.removed == []
            assert diff.changed == ["HomeDomain/Library/Contacts/Contacts.db"]

    def test_diff_combined_changes(self) -> None:
        """Test diff with all three types of changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)

            backup1_dir = parent_dir / "backup1"
            backup2_dir = parent_dir / "backup2"

            files_a = {
                ("HomeDomain", "Library/SMS/sms.db"): ("aaaa0000000000000000000000000000000000"),
                ("HomeDomain", "Library/Contacts/Contacts.db"): (
                    "bbbb1111111111111111111111111111111111"
                ),
                ("HomeDomain", "Library/Mail/mail.db"): ("cccc2222222222222222222222222222222222"),
            }
            files_b = {
                ("HomeDomain", "Library/SMS/sms.db"): (
                    "aaaa0000000000000000000000000000000000"  # Unchanged
                ),
                ("HomeDomain", "Library/Contacts/Contacts.db"): (
                    "dddd3333333333333333333333333333333333"  # Changed
                ),
                ("HomeDomain", "Library/Notes/notes.db"): (
                    "eeee4444444444444444444444444444444444"  # Added
                ),
            }

            create_synthetic_backup(
                backup1_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 15, 10, 0, 0),
                files=files_a,
            )
            create_synthetic_backup(
                backup2_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 16, 10, 0, 0),
                files=files_b,
            )

            discovery = BackupDiscovery()
            backup_a = discovery.validate_backup(backup1_dir)
            backup_b = discovery.validate_backup(backup2_dir)

            manager = BackupManager()
            diff = manager.diff(backup_a, backup_b)

            assert diff.added == ["HomeDomain/Library/Notes/notes.db"]
            assert diff.removed == ["HomeDomain/Library/Mail/mail.db"]
            assert diff.changed == ["HomeDomain/Library/Contacts/Contacts.db"]

    def test_diff_multiple_domains(self) -> None:
        """Test diff with files from multiple domains."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)

            backup1_dir = parent_dir / "backup1"
            backup2_dir = parent_dir / "backup2"

            files_a = {
                ("HomeDomain", "Library/SMS/sms.db"): ("aaaa0000000000000000000000000000000000"),
                ("CameraRollDomain", "Media/DCIM/photo1.jpg"): (
                    "bbbb1111111111111111111111111111111111"
                ),
                ("AppDomain", "App/com.example.app"): ("cccc2222222222222222222222222222222222"),
            }
            files_b = {
                ("HomeDomain", "Library/SMS/sms.db"): ("aaaa0000000000000000000000000000000000"),
                ("CameraRollDomain", "Media/DCIM/photo1.jpg"): (
                    "bbbb1111111111111111111111111111111111"
                ),
                ("CameraRollDomain", "Media/DCIM/photo2.jpg"): (
                    "dddd3333333333333333333333333333333333"
                ),
                ("AppDomain", "App/com.example.app"): (
                    "eeee4444444444444444444444444444444444"  # Changed
                ),
            }

            create_synthetic_backup(
                backup1_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 15, 10, 0, 0),
                files=files_a,
            )
            create_synthetic_backup(
                backup2_dir,
                device_name="Test",
                ios_version="18.3",
                udid="device001" + "0" * 31,
                backup_date=datetime(2026, 3, 16, 10, 0, 0),
                files=files_b,
            )

            discovery = BackupDiscovery()
            backup_a = discovery.validate_backup(backup1_dir)
            backup_b = discovery.validate_backup(backup2_dir)

            manager = BackupManager()
            diff = manager.diff(backup_a, backup_b)

            assert diff.added == ["CameraRollDomain/Media/DCIM/photo2.jpg"]
            assert diff.removed == []
            assert diff.changed == ["AppDomain/App/com.example.app"]

    def test_backup_diff_dataclass_frozen(self) -> None:
        """Verify BackupDiff is immutable (frozen dataclass)."""
        diff = BackupDiff(
            added=["a.txt"],
            removed=["b.txt"],
            changed=["c.txt"],
        )

        # Attempt to modify should raise an error
        with pytest.raises(AttributeError):
            diff.added = []  # type: ignore
