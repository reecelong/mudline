"""Extractor protocol — every domain extractor implements this interface.

Extractors are the bridge between raw source data and the Document model. Each
extractor knows how to parse one specific data domain (messages, photos, etc.)
and yield Document objects for the index layer. Extractors read through a
:class:`~mudline.models.resolver.ResourceResolver`, keeping this protocol
decoupled from any concrete backup format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mudline.models.document import Document
    from mudline.models.resolver import ResourceResolver


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

    def extract(self, resolver: ResourceResolver) -> Iterator[Document]:
        """Extract documents from the source.

        Args:
            resolver: ResourceResolver for the target source, used to
                      map domain + relative paths to actual file locations.

        Yields:
            Document objects with fully populated fields including source provenance.

        Raises:
            FileNotFoundError: If the expected database file is missing from the backup.
            ExtractionError: If the database schema is unexpected or data is corrupt.
        """
        ...

    def can_extract(self, resolver: ResourceResolver) -> bool:
        """Check if this extractor's data source exists in the source.

        Returns:
            True if the required files exist, False otherwise.
            Should not raise exceptions.
        """
        ...
