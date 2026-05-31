"""Tests for ManifestResolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from mudline.exceptions import BackupNotFoundError
from mudline.foundation.manifest import FileRecord, ManifestResolver

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "backup"


class TestManifestResolver:
    """Tests against the synthetic backup fixture."""

    def test_init_valid_backup(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        assert resolver.backup_path == backup_path

    def test_init_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(BackupNotFoundError, match="does not exist"):
            ManifestResolver(tmp_path / "nonexistent")

    def test_init_missing_manifest_db(self, tmp_path: Path) -> None:
        with pytest.raises(BackupNotFoundError, match="Manifest.db not found"):
            ManifestResolver(tmp_path)

    def test_resolve_sms_db(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        path = resolver.resolve("HomeDomain", "Library/SMS/sms.db")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_resolve_contacts_db(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        path = resolver.resolve("HomeDomain", "Library/AddressBook/AddressBook.sqlitedb")
        assert path.exists()

    def test_resolve_call_history(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        path = resolver.resolve("HomeDomain", "Library/CallHistoryDB/CallHistory.storedata")
        assert path.exists()

    def test_resolve_not_in_manifest(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        with pytest.raises(FileNotFoundError, match="not in manifest"):
            resolver.resolve("HomeDomain", "Library/Nonexistent/file.db")

    def test_list_domain(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        files = resolver.list_domain("HomeDomain")
        assert len(files) >= 3  # sms.db, AddressBook, CallHistory
        assert all(isinstance(f, FileRecord) for f in files)
        assert all(f.domain == "HomeDomain" for f in files)

    def test_list_domain_empty(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        files = resolver.list_domain("NonexistentDomain")
        assert files == []

    def test_list_domains(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        domains = resolver.list_domains()
        assert "HomeDomain" in domains
        assert domains == sorted(domains)

    def test_file_exists_true(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        assert resolver.file_exists("HomeDomain", "Library/SMS/sms.db")

    def test_file_exists_false(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        assert not resolver.file_exists("HomeDomain", "Library/Nope/nope.db")

    def test_count(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        assert resolver.count() == 3  # sms.db, AddressBook, CallHistory

    def test_resolve_returns_correct_subdirectory(self, backup_path: Path) -> None:
        """Verify the 2-char prefix subdirectory structure."""
        resolver = ManifestResolver(backup_path)
        path = resolver.resolve("HomeDomain", "Library/SMS/sms.db")
        # Parent should be a 2-char hex prefix directory
        assert len(path.parent.name) == 2

    def test_file_record_fields(self, backup_path: Path) -> None:
        resolver = ManifestResolver(backup_path)
        files = resolver.list_domain("HomeDomain")
        sms_files = [f for f in files if f.relative_path == "Library/SMS/sms.db"]
        assert len(sms_files) == 1
        record = sms_files[0]
        assert record.domain == "HomeDomain"
        assert record.flags == 1
        assert len(record.file_id) == 40  # SHA-1 hex length
