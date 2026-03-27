"""Multi-backup manager — group backups by device and compute diffs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mudline.foundation.discovery import BackupDiscovery, BackupInfo
from mudline.foundation.manifest import ManifestResolver

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackupDiff:
    """Result of comparing two backup snapshots.

    Attributes:
        added: List of relative paths that exist in snapshot_b but not snapshot_a.
        removed: List of relative paths that exist in snapshot_a but not snapshot_b.
        changed: List of relative paths that exist in both but have different file IDs.
    """

    added: list[str]
    removed: list[str]
    changed: list[str]


class BackupManager:
    """Manage multiple iOS backups, grouped by device (UDID).

    This class provides high-level backup operations:
    - Discover and register backups from a directory
    - Group backups by device (UDID)
    - List snapshots in chronological order
    - Compare file sets between backups

    Args:
        discovery: Optional BackupDiscovery instance. If None, creates a new one.
    """

    def __init__(self, discovery: BackupDiscovery | None = None) -> None:
        """Initialize the BackupManager.

        Args:
            discovery: Optional BackupDiscovery instance. If None, creates a new one.
        """
        self.discovery = discovery or BackupDiscovery()
        self._backups: dict[str, list[BackupInfo]] = {}  # UDID -> sorted backups

    def scan(self, backup_dir: Path | None = None) -> None:
        """Discover and register backups from a directory.

        Scans the provided directory (or default iOS backup locations) and groups
        discovered backups by UDID. Backups are sorted chronologically within each device.

        Args:
            backup_dir: Directory to scan for backups. If None, scans default
                       iOS backup locations.
        """
        backups = self.discovery.discover(backup_dir)

        # Group by UDID and sort each group chronologically
        udid_map: dict[str, list[BackupInfo]] = {}
        for backup in backups:
            if backup.udid not in udid_map:
                udid_map[backup.udid] = []
            udid_map[backup.udid].append(backup)

        # Sort each device's backups by date
        for udid, device_backups in udid_map.items():
            device_backups.sort(key=lambda b: b.backup_date)
            self._backups[udid] = device_backups

        logger.debug(
            "BackupManager: discovered %d devices with %d total backups",
            len(self._backups),
            sum(len(b) for b in self._backups.values()),
        )

    def list_devices(self) -> list[str]:
        """Return unique UDIDs of all registered devices.

        Returns:
            List of device UDIDs, sorted alphabetically.
        """
        return sorted(self._backups.keys())

    def list_snapshots(self, udid: str) -> list[BackupInfo]:
        """Return all backups for a device in chronological order.

        Args:
            udid: The device UDID to query.

        Returns:
            List of BackupInfo objects for this device, sorted by backup_date (oldest first).
            Returns empty list if device is not registered.
        """
        return self._backups.get(udid, [])

    def get_latest(self, udid: str) -> BackupInfo | None:
        """Return the most recent backup for a device.

        Args:
            udid: The device UDID to query.

        Returns:
            The BackupInfo with the latest backup_date, or None if device not found.
        """
        snapshots = self.list_snapshots(udid)
        return snapshots[-1] if snapshots else None

    def diff(self, backup_a: BackupInfo, backup_b: BackupInfo) -> BackupDiff:
        """Compare file sets between two backups.

        Compares the file manifests of two backups to identify which files were
        added, removed, or changed. A file is considered "changed" if it exists
        in both backups but has a different file ID (SHA-1 hash).

        Args:
            backup_a: The "before" snapshot.
            backup_b: The "after" snapshot.

        Returns:
            BackupDiff with added/removed/changed file lists.

        Raises:
            BackupNotFoundError: If either backup path is invalid or Manifest.db is missing.
        """
        resolver_a = ManifestResolver(backup_a.path)
        resolver_b = ManifestResolver(backup_b.path)

        # Get all files from both backups, keyed by (domain, relative_path)
        files_a: dict[tuple[str, str], str] = {}  # (domain, path) -> file_id
        files_b: dict[tuple[str, str], str] = {}

        # Collect all files from backup_a
        for domain in resolver_a.list_domains():
            for record in resolver_a.list_domain(domain):
                key = (record.domain, record.relative_path)
                files_a[key] = record.file_id

        # Collect all files from backup_b
        for domain in resolver_b.list_domains():
            for record in resolver_b.list_domain(domain):
                key = (record.domain, record.relative_path)
                files_b[key] = record.file_id

        # Compute sets
        keys_a = set(files_a.keys())
        keys_b = set(files_b.keys())

        added_keys = keys_b - keys_a
        removed_keys = keys_a - keys_b
        common_keys = keys_a & keys_b

        # Format as relative paths (domain/relativePath)
        added = [f"{domain}/{path}" for domain, path in sorted(added_keys)]
        removed = [f"{domain}/{path}" for domain, path in sorted(removed_keys)]

        # Files that changed (exist in both but different file_id)
        changed = [
            f"{domain}/{path}"
            for domain, path in sorted(common_keys)
            if files_a[(domain, path)] != files_b[(domain, path)]
        ]

        return BackupDiff(added=added, removed=removed, changed=changed)
