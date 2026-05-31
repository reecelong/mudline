"""ResourceResolver protocol — the minimal file-resolution surface extractors need.

This is the engine-side abstraction that decouples the :class:`~mudline.models.extractor.Extractor`
protocol from any concrete backup format. Concrete resolvers (e.g. the iOS
``ManifestResolver``) implement a richer interface, but extractors depend only on
the three members declared here, so the ``mudline.models`` package stays free of
``mudline.foundation`` imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class ResourceResolver(Protocol):
    """Maps a logical ``(domain, relative_path)`` address to a file on disk.

    Any object providing these members satisfies the protocol structurally —
    no inheritance required. The iOS ``ManifestResolver`` conforms as-is.
    """

    @property
    def backup_path(self) -> Path:
        """Root directory of the source the documents are extracted from."""
        ...

    def resolve(self, domain: str, relative_path: str) -> Path:
        """Resolve a logical address to a usable file path on disk.

        Raises:
            FileNotFoundError: If the address cannot be resolved to a real file.
        """
        ...

    def file_exists(self, domain: str, relative_path: str) -> bool:
        """Return whether a logical address exists, without touching disk."""
        ...
