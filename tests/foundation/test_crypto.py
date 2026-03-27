"""Tests for KeybagDecryptor and encrypted backup support."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from mudline.exceptions import DecryptionError
from mudline.foundation.crypto import KeybagDecryptor
from mudline.foundation.manifest import ManifestResolver


class TestKeybagDecryptor:
    """Tests for KeybagDecryptor class."""

    def test_init_success(self, tmp_path: Path) -> None:
        """Test successful initialization with mocked iOSbackup."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            mock_ios.return_value = mock_instance

            decryptor = KeybagDecryptor(backup_path, "password123")
            assert decryptor._backup_instance is not None
            assert decryptor._password == "password123"
            assert decryptor._backup_path == backup_path

            # Verify iOSbackup was initialized with correct params
            mock_ios.assert_called_once()
            call_kwargs = mock_ios.call_args[1]
            assert call_kwargs["udid"] == "00008140-ABC123DEF456"
            assert call_kwargs["cleartextpassword"] == "password123"

            decryptor.close()

    def test_init_invalid_password(self, tmp_path: Path) -> None:
        """Test initialization fails with invalid password."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        with patch(
            "mudline.foundation.crypto.iOSbackup.iOSbackup"
        ) as mock_ios:
            # Simulate the iOSbackup library raising an exception with password error
            mock_ios.side_effect = Exception("Invalid password")

            with pytest.raises(DecryptionError, match="Invalid password"):
                KeybagDecryptor(backup_path, "wrongpassword")

    def test_decrypt_file_success(self, tmp_path: Path) -> None:
        """Test successful file decryption."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        decrypted_file = tmp_path / "decrypted_sms.db"
        decrypted_file.write_bytes(b"decrypted data")

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            mock_instance.getFileDecryptedCopy.return_value = {
                "decryptedFilePath": str(decrypted_file)
            }
            mock_ios.return_value = mock_instance

            decryptor = KeybagDecryptor(backup_path, "password123")
            result = decryptor.decrypt_file(
                "HomeDomain", "Library/SMS/sms.db"
            )

            assert result == decrypted_file
            mock_instance.getFileDecryptedCopy.assert_called_once_with(
                relativePath="Library/SMS/sms.db",
                temporary=True,
            )
            decryptor.close()

    def test_decrypt_file_not_found(self, tmp_path: Path) -> None:
        """Test decryption fails when file doesn't exist."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            # Simulate FileNotFound exception from iOSbackup
            mock_instance.getFileDecryptedCopy.side_effect = FileNotFoundError(
                "Can't find backup entry for relative path"
            )
            mock_ios.return_value = mock_instance

            decryptor = KeybagDecryptor(backup_path, "password123")

            with pytest.raises(FileNotFoundError, match="not found in backup"):
                decryptor.decrypt_file(
                    "HomeDomain", "Library/Nonexistent/file.db"
                )

            decryptor.close()

    def test_decrypt_file_no_decrypted_path(self, tmp_path: Path) -> None:
        """Test decryption fails when response lacks decrypted path."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            mock_instance.getFileDecryptedCopy.return_value = {}
            mock_ios.return_value = mock_instance

            decryptor = KeybagDecryptor(backup_path, "password123")

            with pytest.raises(DecryptionError, match="no decrypted path"):
                decryptor.decrypt_file("HomeDomain", "Library/SMS/sms.db")

            decryptor.close()

    def test_decrypt_data_success(self, tmp_path: Path) -> None:
        """Test successful in-memory file decryption."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        expected_data = b"decrypted database content"

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            mock_instance.getRelativePathDecryptedData.return_value = (
                {"size": len(expected_data)},
                expected_data,
            )
            mock_ios.return_value = mock_instance

            decryptor = KeybagDecryptor(backup_path, "password123")
            result = decryptor.decrypt_data(
                "HomeDomain", "Library/SMS/sms.db"
            )

            assert result == expected_data
            mock_instance.getRelativePathDecryptedData.assert_called_once_with(
                "Library/SMS/sms.db"
            )
            decryptor.close()

    def test_decrypt_data_not_found(self, tmp_path: Path) -> None:
        """Test in-memory decryption fails when file doesn't exist."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            mock_instance.getRelativePathDecryptedData.side_effect = FileNotFoundError(
                "Can't find backup entry for relative path"
            )
            mock_ios.return_value = mock_instance

            decryptor = KeybagDecryptor(backup_path, "password123")

            with pytest.raises(FileNotFoundError, match="not found in backup"):
                decryptor.decrypt_data(
                    "HomeDomain", "Library/Nonexistent/file.db"
                )

            decryptor.close()

    def test_close_closes_backup_instance(self, tmp_path: Path) -> None:
        """Test that close() properly closes the backup instance."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            mock_ios.return_value = mock_instance

            decryptor = KeybagDecryptor(backup_path, "password123")
            assert decryptor._backup_instance is not None

            decryptor.close()

            # Verify close was called on the backup instance
            mock_instance.close.assert_called_once()
            # Verify the instance is nulled out
            assert decryptor._backup_instance is None

    def test_close_idempotent(self, tmp_path: Path) -> None:
        """Test that close() can be called multiple times safely."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            mock_ios.return_value = mock_instance

            decryptor = KeybagDecryptor(backup_path, "password123")
            decryptor.close()
            decryptor.close()  # Should not raise

            # close() should only be called once
            mock_instance.close.assert_called_once()

    def test_context_manager_enters_and_exits(self, tmp_path: Path) -> None:
        """Test KeybagDecryptor as a context manager."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            mock_ios.return_value = mock_instance

            with KeybagDecryptor(backup_path, "password123") as decryptor:
                assert decryptor._backup_instance is not None
                mock_instance.close.assert_not_called()

            # After exiting context, close should be called
            mock_instance.close.assert_called_once()

    def test_decrypt_after_close_raises(self, tmp_path: Path) -> None:
        """Test that decryption fails after close()."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            mock_ios.return_value = mock_instance

            decryptor = KeybagDecryptor(backup_path, "password123")
            decryptor.close()

            with pytest.raises(DecryptionError, match="Decryptor is closed"):
                decryptor.decrypt_file("HomeDomain", "Library/SMS/sms.db")

    def test_decrypt_data_after_close_raises(self, tmp_path: Path) -> None:
        """Test that in-memory decryption fails after close()."""
        backup_path = tmp_path / "00008140-ABC123DEF456"
        backup_path.mkdir()

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup") as mock_ios:
            mock_instance = MagicMock()
            mock_ios.return_value = mock_instance

            decryptor = KeybagDecryptor(backup_path, "password123")
            decryptor.close()

            with pytest.raises(DecryptionError, match="Decryptor is closed"):
                decryptor.decrypt_data("HomeDomain", "Library/SMS/sms.db")


class TestManifestResolverWithDecryptor:
    """Tests for ManifestResolver with encrypted backup support."""

    def test_init_without_decryptor(self, backup_path: Path) -> None:
        """Test ManifestResolver can be initialized without decryptor."""
        resolver = ManifestResolver(backup_path)
        assert resolver._decryptor is None

    def test_init_with_decryptor(
        self, backup_path: Path, tmp_path: Path
    ) -> None:
        """Test ManifestResolver can be initialized with decryptor."""
        encrypted_backup_path = tmp_path / "00008140-ENCRYPTED"
        encrypted_backup_path.mkdir()

        # Copy Manifest.db from the fixture backup
        import shutil

        manifest_src = backup_path / "Manifest.db"
        manifest_dst = encrypted_backup_path / "Manifest.db"
        shutil.copy(manifest_src, manifest_dst)

        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup"):
            decryptor = Mock()
            resolver = ManifestResolver(encrypted_backup_path, decryptor)
            assert resolver._decryptor is decryptor

    def test_resolve_decrypted_without_decryptor(
        self, backup_path: Path
    ) -> None:
        """Test resolve_decrypted fails when no decryptor is configured."""
        resolver = ManifestResolver(backup_path)

        with pytest.raises(
            RuntimeError, match="no decryptor configured"
        ):
            resolver.resolve_decrypted("HomeDomain", "Library/SMS/sms.db")

    def test_resolve_decrypted_file_not_in_manifest(
        self, backup_path: Path, tmp_path: Path
    ) -> None:
        """Test resolve_decrypted fails for files not in manifest."""
        with patch("mudline.foundation.crypto.iOSbackup.iOSbackup"):
            decryptor = Mock()
            resolver = ManifestResolver(backup_path, decryptor)

            with pytest.raises(
                FileNotFoundError, match="not in manifest"
            ):
                resolver.resolve_decrypted(
                    "HomeDomain", "Library/Nonexistent/file.db"
                )

    def test_resolve_decrypted_delegates_to_decryptor(
        self, backup_path: Path, tmp_path: Path
    ) -> None:
        """Test resolve_decrypted delegates to the decryptor."""
        expected_path = tmp_path / "decrypted_sms.db"
        expected_path.write_bytes(b"data")

        decryptor = Mock()
        decryptor.decrypt_file.return_value = expected_path

        resolver = ManifestResolver(backup_path, decryptor)
        result = resolver.resolve_decrypted(
            "HomeDomain", "Library/SMS/sms.db"
        )

        assert result == expected_path
        decryptor.decrypt_file.assert_called_once_with(
            "HomeDomain", "Library/SMS/sms.db"
        )

    def test_resolve_still_works_with_decryptor(
        self, backup_path: Path
    ) -> None:
        """Test that resolve() still works when decryptor is present."""
        decryptor = Mock()
        resolver = ManifestResolver(backup_path, decryptor)

        # Should still be able to resolve unencrypted files
        path = resolver.resolve("HomeDomain", "Library/SMS/sms.db")
        assert path.exists()
        # Decryptor should not have been called
        decryptor.decrypt_file.assert_not_called()

    def test_resolve_decrypted_propagates_decryption_error(
        self, backup_path: Path, tmp_path: Path
    ) -> None:
        """Test that decryption errors are propagated."""
        from mudline.exceptions import DecryptionError

        decryptor = Mock()
        decryptor.decrypt_file.side_effect = DecryptionError(
            "Failed to decrypt"
        )

        resolver = ManifestResolver(backup_path, decryptor)

        with pytest.raises(DecryptionError, match="Failed to decrypt"):
            resolver.resolve_decrypted("HomeDomain", "Library/SMS/sms.db")
