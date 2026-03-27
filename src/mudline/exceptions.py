"""Mudline custom exceptions."""


class RecallError(Exception):
    """Base exception for all Recall errors."""


class BackupNotFoundError(RecallError):
    """Raised when a backup path doesn't contain a valid iOS backup."""


class DecryptionError(RecallError):
    """Raised when backup decryption fails (wrong password, corrupt keybag)."""


class ExtractionError(RecallError):
    """Raised when an extractor encounters unexpected schema or corrupt data."""


class SearchError(RecallError):
    """Raised when the search infrastructure is unavailable or query fails."""


class LLMError(RecallError):
    """Raised when the LLM provider returns an error or is unavailable."""
