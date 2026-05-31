"""Encrypted backup support — decrypt iOS backup files using keybag.

Wraps the iOSbackup library to provide a unified interface for decrypting
files from encrypted iOS backups. Handles password validation, temporary
file management, and error conversion to Mudline exception types.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import iOSbackup
except ModuleNotFoundError:  # pragma: no cover - only hit without the 'ios' extra
    iOSbackup = None  # noqa: N816 - must match the library module name

from mudline.exceptions import DecryptionError

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)


class KeybagDecryptor:
    """Decrypt files from an encrypted iOS backup.

    This class wraps the iOSbackup library to provide decryption of individual
    files from encrypted iOS backups. It manages the iOSbackup instance lifecycle
    and converts library exceptions to Mudline exception types.

    The decryptor can be used as a context manager for automatic cleanup.

    Args:
        backup_path: Path to the root of the encrypted iOS backup directory.
        password: Cleartext password for the backup.

    Raises:
        DecryptionError: If the backup is invalid, password is wrong, or
            keybag parsing fails.
    """

    def __init__(self, backup_path: Path, password: str) -> None:
        """Initialize the decryptor and validate the backup password.

        Args:
            backup_path: Path to the encrypted iOS backup directory.
            password: Cleartext password to decrypt the backup.

        Raises:
            DecryptionError: If password is incorrect or keybag is corrupted.
        """
        if iOSbackup is None:
            raise DecryptionError(
                "Encrypted backup support requires the 'ios' extra. "
                "Install it with: pip install 'mudline[ios]'"
            )

        self._backup_path = backup_path
        self._password = password
        self._backup_instance: iOSbackup.iOSbackup | None = None
        self._temp_dir = tempfile.TemporaryDirectory()

        try:
            # Extract UDID from the backup directory name
            # iOS backups are typically stored as UUIDs (e.g., 00008140-...)
            udid = backup_path.name

            # Initialize iOSbackup with the provided credentials
            self._backup_instance = iOSbackup.iOSbackup(
                udid=udid,
                cleartextpassword=password,
                backuproot=str(backup_path.parent),
            )

            logger.debug("KeybagDecryptor initialized for backup %s", backup_path)

        except Exception as e:
            self._cleanup()
            # Check if this is a password error
            if "password" in str(e).lower():
                raise DecryptionError(f"Invalid password for backup {backup_path}") from e
            raise DecryptionError(
                f"Failed to initialize backup decryption for {backup_path}: {e}"
            ) from e

    def decrypt_file(self, domain: str, relative_path: str) -> Path:
        """Decrypt a single file to a temporary location.

        Returns a path to the decrypted file in a temporary directory.
        The file persists until the decryptor is closed or the context
        manager exits.

        Args:
            domain: iOS backup domain (e.g., "HomeDomain").
            relative_path: Path within the domain (e.g., "Library/SMS/sms.db").

        Returns:
            Path to the decrypted file (in a temporary directory).

        Raises:
            FileNotFoundError: If the file doesn't exist in the backup.
            DecryptionError: If decryption fails.
        """
        if self._backup_instance is None:
            raise DecryptionError("Decryptor is closed")

        try:
            result = self._backup_instance.getFileDecryptedCopy(
                relativePath=relative_path,
                temporary=True,
            )
            decrypted_path = result.get("decryptedFilePath")
            if not decrypted_path:
                raise DecryptionError(
                    f"Failed to decrypt {domain}/{relative_path}: no decrypted path returned"
                )
            return Path(decrypted_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found in backup: {domain}/{relative_path}") from e
        except DecryptionError:
            raise
        except Exception as e:
            raise DecryptionError(f"Failed to decrypt {domain}/{relative_path}: {e}") from e

    def decrypt_data(self, domain: str, relative_path: str) -> bytes:
        """Decrypt a single file to bytes in memory.

        Suitable for small files. For large files, prefer decrypt_file()
        to avoid loading the entire file into memory.

        Args:
            domain: iOS backup domain (e.g., "HomeDomain").
            relative_path: Path within the domain (e.g., "Library/SMS/sms.db").

        Returns:
            Decrypted file contents as bytes.

        Raises:
            FileNotFoundError: If the file doesn't exist in the backup.
            DecryptionError: If decryption fails.
        """
        if self._backup_instance is None:
            raise DecryptionError("Decryptor is closed")

        try:
            # getRelativePathDecryptedData returns (info_dict, bytes)
            _info, data = self._backup_instance.getRelativePathDecryptedData(relative_path)
            return data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found in backup: {domain}/{relative_path}") from e
        except DecryptionError:
            raise
        except Exception as e:
            raise DecryptionError(f"Failed to decrypt {domain}/{relative_path}: {e}") from e

    def get_manifest_db_path(self) -> Path:
        """Get the path to a decrypted copy of Manifest.db.

        For encrypted backups, iOSbackup decrypts Manifest.db to a temporary
        file during initialization. This method returns that path so
        ManifestResolver can query the decrypted manifest.

        Returns:
            Path to the decrypted Manifest.db.

        Raises:
            DecryptionError: If the backup instance is closed or manifest
                decryption failed.
        """
        if self._backup_instance is None:
            raise DecryptionError("Decryptor is closed")

        manifest_db = getattr(self._backup_instance, "manifestDB", None)
        if manifest_db is None:
            raise DecryptionError("Decrypted Manifest.db not available")

        return Path(manifest_db)

    def close(self) -> None:
        """Close the backup connection and clean up temporary files.

        Safe to call multiple times. After closing, the decryptor cannot
        be used for further decryption operations.
        """
        if self._backup_instance is not None:
            try:
                self._backup_instance.close()
            except Exception as e:
                logger.warning("Error closing backup instance: %s", e)
            self._backup_instance = None

        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up temporary directory if created."""
        try:
            self._temp_dir.cleanup()
        except Exception as e:
            logger.warning("Error cleaning up temp directory: %s", e)

    def __enter__(self) -> KeybagDecryptor:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and clean up resources."""
        self.close()
