"""ManifestResolver — map iOS backup SHA-1 file hashes to domain + relative path.

iOS backups store files in a flat directory structure using SHA-1 hashes as filenames,
organized into 256 two-character prefix subdirectories. Manifest.db maps
(domain, relativePath) → fileID (the SHA-1 hash), allowing us to find actual files.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mudline.exceptions import BackupNotFoundError

if TYPE_CHECKING:
    from pathlib import Path

    from mudline.foundation.crypto import KeybagDecryptor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileRecord:
    """A single file entry from Manifest.db.

    Attributes:
        file_id: SHA-1 hash used as the filename in the backup.
        domain: iOS backup domain (e.g., "HomeDomain", "CameraRollDomain").
        relative_path: Path within the domain (e.g., "Library/SMS/sms.db").
        flags: File type flags (1=file, 2=directory, 4=symlink).
    """

    file_id: str
    domain: str
    relative_path: str
    flags: int


class ManifestResolver:
    """Resolve iOS backup file paths via Manifest.db.

    iOS backups store all files as SHA-1 hashes in two-character prefix
    subdirectories. This class parses Manifest.db to map logical paths
    (domain + relativePath) to actual file locations on disk.

    Supports both unencrypted and encrypted backups. For encrypted backups,
    provide a KeybagDecryptor instance to enable decryption of individual files.

    Args:
        backup_path: Path to the root of the iOS backup directory.
        decryptor: Optional KeybagDecryptor for encrypted backups.

    Raises:
        BackupNotFoundError: If backup_path doesn't exist or Manifest.db is missing.
    """

    def __init__(
        self, backup_path: Path, decryptor: KeybagDecryptor | None = None
    ) -> None:
        self._backup_path = backup_path
        self._decryptor = decryptor
        self._encrypted = False

        if not backup_path.is_dir():
            raise BackupNotFoundError(f"Backup directory does not exist: {backup_path}")

        raw_db_path = backup_path / "Manifest.db"
        if not raw_db_path.exists():
            raise BackupNotFoundError(
                f"Manifest.db not found in backup: {backup_path}"
            )

        # Try opening the raw Manifest.db first. If it's encrypted (not valid
        # SQLite), fall back to the decryptor's decrypted copy.
        try:
            conn = sqlite3.connect(
                f"file:{raw_db_path}?mode=ro&immutable=1", uri=True
            )
            conn.execute("SELECT 1 FROM Files LIMIT 1")
            conn.close()
            self._db_path = raw_db_path
        except sqlite3.Error:
            if decryptor is None:
                raise BackupNotFoundError(
                    f"Manifest.db at {raw_db_path} is encrypted. "
                    "Provide a KeybagDecryptor to read encrypted backups."
                )
            # Use the decryptor's decrypted manifest
            self._db_path = decryptor.get_manifest_db_path()
            self._encrypted = True
            # Validate the decrypted copy
            try:
                conn = sqlite3.connect(
                    f"file:{self._db_path}?mode=ro", uri=True
                )
                conn.execute("SELECT 1 FROM Files LIMIT 1")
                conn.close()
            except sqlite3.Error as e:
                raise BackupNotFoundError(
                    f"Decrypted Manifest.db is not valid: {e}"
                ) from e

        logger.debug(
            "ManifestResolver initialized for %s (encrypted=%s)",
            backup_path,
            self._encrypted,
        )

    @property
    def backup_path(self) -> Path:
        """The root backup directory."""
        return self._backup_path

    def _connect(self) -> sqlite3.Connection:
        """Open a read-only connection to the manifest database.

        Uses immutable mode for raw (unencrypted) manifests and standard
        read-only mode for decrypted temp copies.
        """
        if self._encrypted:
            return sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        return sqlite3.connect(
            f"file:{self._db_path}?mode=ro&immutable=1", uri=True
        )

    def resolve(self, domain: str, relative_path: str) -> Path:
        """Resolve a domain + relative path to a usable file on disk.

        For encrypted backups, this transparently returns a decrypted temporary
        copy of the file. Callers don't need to distinguish between encrypted
        and unencrypted backups.

        Args:
            domain: iOS backup domain (e.g., "HomeDomain").
            relative_path: Path within the domain (e.g., "Library/SMS/sms.db").

        Returns:
            Path to the file (raw for unencrypted, decrypted temp for encrypted).

        Raises:
            FileNotFoundError: If the file is not in Manifest.db or doesn't
                exist on disk.
            DecryptionError: If decryption fails (encrypted backups only).
        """
        if self._encrypted:
            return self.resolve_decrypted(domain, relative_path)

        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT fileID FROM Files WHERE domain = ? AND relativePath = ?",
                (domain, relative_path),
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            raise FileNotFoundError(
                f"File not in manifest: {domain}/{relative_path}"
            )

        file_id: str = row[0]
        file_path = self._backup_path / file_id[:2] / file_id

        if not file_path.exists():
            raise FileNotFoundError(
                f"File in manifest but missing from disk: {domain}/{relative_path} "
                f"(expected at {file_path})"
            )

        return file_path

    def resolve_decrypted(self, domain: str, relative_path: str) -> Path:
        """Resolve and decrypt a file from an encrypted backup.

        Returns a path to a decrypted temporary copy of the file. The temp
        file persists until the associated KeybagDecryptor is closed.

        Args:
            domain: iOS backup domain (e.g., "HomeDomain").
            relative_path: Path within the domain (e.g., "Library/SMS/sms.db").

        Returns:
            Path to the decrypted file (in a temporary directory).

        Raises:
            RuntimeError: If no decryptor is configured.
            FileNotFoundError: If the file is not in the manifest.
            DecryptionError: If decryption fails.
        """
        if self._decryptor is None:
            raise RuntimeError(
                "Cannot resolve decrypted file: no decryptor configured. "
                "Pass a KeybagDecryptor to ManifestResolver.__init__() for "
                "encrypted backups."
            )

        # Verify the file exists in the manifest
        if not self.file_exists(domain, relative_path):
            raise FileNotFoundError(
                f"File not in manifest: {domain}/{relative_path}"
            )

        return self._decryptor.decrypt_file(domain, relative_path)

    def list_domain(self, domain: str) -> list[FileRecord]:
        """List all files in a specific iOS backup domain.

        Args:
            domain: iOS backup domain to list (e.g., "HomeDomain").

        Returns:
            List of FileRecord objects for all files in the domain.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT fileID, domain, relativePath, flags FROM Files "
                "WHERE domain = ? ORDER BY relativePath",
                (domain,),
            )
            return [
                FileRecord(
                    file_id=row[0],
                    domain=row[1],
                    relative_path=row[2],
                    flags=row[3],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def list_domains(self) -> list[str]:
        """List all unique domains in the backup.

        Returns:
            Sorted list of domain names.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT DISTINCT domain FROM Files ORDER BY domain"
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def file_exists(self, domain: str, relative_path: str) -> bool:
        """Check if a file exists in the manifest (without checking disk).

        Args:
            domain: iOS backup domain.
            relative_path: Path within the domain.

        Returns:
            True if the file is in Manifest.db.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT 1 FROM Files WHERE domain = ? AND relativePath = ? LIMIT 1",
                (domain, relative_path),
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def count(self) -> int:
        """Count total number of files in the manifest.

        Returns:
            Total file count.
        """
        conn = self._connect()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM Files")
            row = cursor.fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
