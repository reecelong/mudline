"""Extractor registry — auto-discovery and management of domain extractors.

This module provides a registry for discovering and instantiating extractors.
The default_registry is pre-populated with all built-in extractors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mudline.models.extractor import Extractor

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """Registry for discovering and managing extractors.

    Provides registration, lookup, and instantiation of extractor classes.
    """

    def __init__(self) -> None:
        """Initialize the registry with no extractors."""
        self._extractors: dict[str, type] = {}

    def register(self, name: str, extractor_cls: type) -> None:
        """Register an extractor class.

        Args:
            name: Unique name for the extractor (e.g., "messages").
            extractor_cls: The extractor class to register.
        """
        self._extractors[name] = extractor_cls
        logger.debug(f"Registered extractor: {name}")

    def get(self, name: str) -> type:
        """Get a registered extractor class.

        Args:
            name: The name of the extractor to retrieve.

        Returns:
            The extractor class.

        Raises:
            KeyError: If the extractor is not registered.
        """
        if name not in self._extractors:
            raise KeyError(f"Extractor '{name}' not found in registry")
        return self._extractors[name]

    def list(self) -> list[str]:
        """List all registered extractor names.

        Returns:
            List of registered extractor names in sorted order.
        """
        return sorted(self._extractors.keys())

    def create_all(self) -> list[Extractor]:
        """Create instances of all registered extractors.

        Returns:
            List of instantiated extractors.
        """
        extractors = []
        for name in self.list():
            extractor_cls = self._extractors[name]
            try:
                extractor = extractor_cls()
                extractors.append(extractor)
                logger.debug(f"Created extractor instance: {name}")
            except Exception as e:
                logger.error(f"Failed to create extractor {name}: {e}")
        return extractors


# Module-level default registry with all built-in extractors pre-registered
default_registry = ExtractorRegistry()

# Register all built-in extractors
try:
    from mudline.extractors.calendar import CalendarExtractor
    default_registry.register("calendar", CalendarExtractor)
except ImportError as e:
    logger.warning(f"Failed to import CalendarExtractor: {e}")

try:
    from mudline.extractors.calls import CallHistoryExtractor
    default_registry.register("calls", CallHistoryExtractor)
except ImportError as e:
    logger.warning(f"Failed to import CallHistoryExtractor: {e}")

try:
    from mudline.extractors.contacts import ContactExtractor
    default_registry.register("contacts", ContactExtractor)
except ImportError as e:
    logger.warning(f"Failed to import ContactExtractor: {e}")

try:
    from mudline.extractors.messages import MessageExtractor
    default_registry.register("messages", MessageExtractor)
except ImportError as e:
    logger.warning(f"Failed to import MessageExtractor: {e}")

try:
    from mudline.extractors.notes import NoteExtractor
    default_registry.register("notes", NoteExtractor)
except ImportError as e:
    logger.warning(f"Failed to import NoteExtractor: {e}")

try:
    from mudline.extractors.photos import PhotoExtractor
    default_registry.register("photos", PhotoExtractor)
except ImportError as e:
    logger.warning(f"Failed to import PhotoExtractor: {e}")

try:
    from mudline.extractors.safari import SafariExtractor
    default_registry.register("safari", SafariExtractor)
except ImportError as e:
    logger.warning(f"Failed to import SafariExtractor: {e}")

try:
    from mudline.extractors.voicemail import VoicemailExtractor
    default_registry.register("voicemail", VoicemailExtractor)
except ImportError as e:
    logger.warning(f"Failed to import VoicemailExtractor: {e}")
