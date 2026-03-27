"""Extractor protocol — every domain extractor implements this interface.

Extractors are the bridge between raw iOS backup data and the Document model.
Each extractor knows how to parse one specific iOS data domain (messages, photos, etc.)
and yield Document objects for the index layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mudline.foundation.manifest import ManifestResolver
    from mudline.models.document import Document


@runtime_checkable
class Extractor(Protocol):
    """Protocol that all domain extractors must implement.

    Attributes:
        domain: The iOS backup domain this extractor reads from (e.g., "HomeDomain").
        data_type: The DocumentType this extractor produces (e.g., "message").
    """

    @property
    def domain(self) -> str:
        """The iOS backup domain (e.g., 'HomeDomain', 'CameraRollDomain')."""
        ...

    @property
    def data_type(self) -> str:
        """The document type string (should match a DocumentType value)."""
        ...

    def extract(self, resolver: ManifestResolver) -> Iterator[Document]:
        """Extract documents from the backup.

        Args:
            resolver: ManifestResolver for the target backup, used to
                      map domain + relative paths to actual file locations.

        Yields:
            Document objects with fully populated fields including source provenance.

        Raises:
            FileNotFoundError: If the expected database file is missing from the backup.
            ExtractionError: If the database schema is unexpected or data is corrupt.
        """
        ...

    def can_extract(self, resolver: ManifestResolver) -> bool:
        """Check if this extractor's data source exists in the backup.

        Returns:
            True if the required files exist, False otherwise.
            Should not raise exceptions.
        """
        ...
