"""Backup discovery — scan filesystem for valid iOS backup directories.

Identifies device name, iOS version, UDID, backup date, and encryption status
by parsing Info.plist, Status.plist, and Manifest.plist from backup directories.
"""

from __future__ import annotations

import logging
import plistlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mudline.exceptions import BackupNotFoundError

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackupInfo:
    """Metadata extracted from an iOS backup directory.

    Attributes:
        device_name: Display name of the iOS device (e.g., "Jane's iPhone").
        ios_version: iOS/iPadOS version string (e.g., "18.3").
        udid: Unique Device Identifier (40-character hex string).
        backup_date: Timestamp when the backup was created.
        is_encrypted: Whether the backup is encrypted (requires password).
        path: Absolute path to the backup directory.
    """

    device_name: str
    ios_version: str
    udid: str
    backup_date: datetime
    is_encrypted: bool
    path: Path


class BackupDiscovery:
    """Scan filesystem for valid iOS backup directories and extract metadata."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the backup discovery scanner.

        Args:
            verbose: If True, log detailed information about discovered backups.
        """
        self.verbose = verbose

    def discover(self, backup_dir: Path | None = None) -> list[BackupInfo]:
        """Discover all valid iOS backups in a directory or default locations.

        Scans the provided directory or default iOS backup locations for valid
        backup subdirectories. A valid backup must contain Info.plist.

        Args:
            backup_dir: Scan this directory for backups. If None, scan default
                       locations (~/.PluginKit/DB, ~/Library/MobileSync/Backup/).

        Returns:
            List of BackupInfo objects for each discovered backup.

        Raises:
            BackupNotFoundError: If backup_dir is specified but is not a directory.
        """
        backups: list[BackupInfo] = []

        if backup_dir is not None:
            if not backup_dir.is_dir():
                raise BackupNotFoundError(f"Not a directory: {backup_dir}")
            backups.extend(self._scan_directory(backup_dir))
        else:
            # Scan default macOS locations
            default_locations = [
                Path.home() / "Library" / "Application Support" / "MobileSync" / "Backup",
                Path.home() / "Library" / "MobileSync" / "Backup",
            ]
            for location in default_locations:
                if location.is_dir():
                    backups.extend(self._scan_directory(location))

        return backups

    def _scan_directory(self, directory: Path) -> list[BackupInfo]:
        """Scan a directory for backup subdirectories.

        Args:
            directory: Directory containing backup subdirectories.

        Returns:
            List of BackupInfo objects found in this directory.
        """
        backups: list[BackupInfo] = []

        try:
            for item in directory.iterdir():
                if not item.is_dir():
                    continue

                backup_info = self._parse_backup(item)
                if backup_info is not None:
                    backups.append(backup_info)
                    if self.verbose:
                        logger.info(
                            "Discovered backup: %s (UDID: %s, iOS %s, encrypted=%s)",
                            backup_info.device_name,
                            backup_info.udid,
                            backup_info.ios_version,
                            backup_info.is_encrypted,
                        )
        except (OSError, PermissionError) as e:
            logger.warning("Failed to scan directory %s: %s", directory, e)

        return backups

    def _parse_backup(self, backup_path: Path) -> BackupInfo | None:
        """Parse Info.plist and Manifest.plist from a backup directory.

        Args:
            backup_path: Path to a potential backup directory.

        Returns:
            BackupInfo if this is a valid iOS backup, None otherwise.
        """
        info_plist_path = backup_path / "Info.plist"
        manifest_plist_path = backup_path / "Manifest.plist"

        # Must have Info.plist to be a valid backup
        if not info_plist_path.exists():
            return None

        try:
            # Parse Info.plist
            with open(info_plist_path, "rb") as f:
                info_plist = plistlib.load(f)

            # Extract device metadata
            device_name = info_plist.get("Device Name", "Unknown Device")
            ios_version = info_plist.get("Product Version", "Unknown")
            udid = info_plist.get("Unique Identifier", "")
            backup_date = info_plist.get("Last Backup Date")

            # Validate critical fields
            if not udid:
                logger.warning("Backup at %s missing Unique Identifier", backup_path)
                return None

            if backup_date is None:
                logger.warning("Backup at %s missing Last Backup Date", backup_path)
                return None

            # Parse Manifest.plist for encryption status
            is_encrypted = False
            if manifest_plist_path.exists():
                try:
                    with open(manifest_plist_path, "rb") as f:
                        manifest_plist = plistlib.load(f)
                    is_encrypted = manifest_plist.get("IsEncrypted", False)
                except Exception as e:
                    logger.warning(
                        "Failed to parse Manifest.plist at %s: %s", backup_path, e
                    )
                    # Continue with is_encrypted=False if we can't parse it

            return BackupInfo(
                device_name=device_name,
                ios_version=ios_version,
                udid=udid,
                backup_date=backup_date,
                is_encrypted=is_encrypted,
                path=backup_path,
            )

        except Exception as e:
            logger.warning("Failed to parse backup at %s: %s", backup_path, e)
            return None

    def validate_backup(self, backup_path: Path) -> BackupInfo:
        """Validate and parse a single backup directory.

        Strictly validates that the path contains a valid backup. Used when
        a specific backup path is provided by the user.

        Args:
            backup_path: Path to validate.

        Returns:
            BackupInfo for this backup.

        Raises:
            BackupNotFoundError: If the path does not contain a valid backup.
        """
        if not backup_path.is_dir():
            raise BackupNotFoundError(f"Not a directory: {backup_path}")

        backup_info = self._parse_backup(backup_path)
        if backup_info is None:
            raise BackupNotFoundError(
                f"No valid iOS backup found at {backup_path} "
                "(missing or invalid Info.plist)"
            )

        return backup_info
