"""Contact resolution index — maps handles to names with fuzzy matching.

This module provides a contact index that maps phone numbers, emails, and iMessage
handles to resolved display names with fuzzy matching support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class ContactIndex:
    """Index for resolving contact handles to names with fuzzy matching.

    Maintains bidirectional mapping between contact names and their associated
    handles (phone numbers, emails, iMessage handles).
    """

    def __init__(self) -> None:
        """Initialize the contact index."""
        self._name_to_handles: dict[str, set[str]] = {}
        self._handle_to_name: dict[str, str] = {}

    def load_from_documents(self, docs: Iterable) -> None:  # type: ignore[override]
        """Load contact information from Document objects.

        Extracts contact data from CONTACT type documents and populates
        the name-to-handles and handle-to-name mappings.

        Args:
            docs: Iterable of Document objects (typically from extractors).
        """
        from mudline.models.document import DocumentType

        for doc in docs:
            if doc.type != DocumentType.CONTACT:
                continue

            # Extract name from document text
            # Text format: "FirstName LastName — org — phone — email"
            name = doc.text.split(" — ")[0].strip() if " — " in doc.text else doc.text.strip()

            if not name:
                continue

            # Normalize name to canonical form (title case)
            canonical_name = name.title()

            # Extract handles from metadata
            phones = doc.metadata.get("phones", [])
            emails = doc.metadata.get("emails", [])
            handles = doc.metadata.get("handles", [])

            # Combine all handles
            all_handles = set()
            all_handles.update(self.normalize_handle(h) for h in phones)
            all_handles.update(self.normalize_handle(h) for h in emails)
            all_handles.update(self.normalize_handle(h) for h in handles)

            if not all_handles:
                continue

            # Store mappings
            if canonical_name not in self._name_to_handles:
                self._name_to_handles[canonical_name] = set()
            self._name_to_handles[canonical_name].update(all_handles)

            # Store reverse mappings
            for handle in all_handles:
                self._handle_to_name[handle] = canonical_name

            logger.debug(f"Loaded contact {canonical_name} with {len(all_handles)} handles")

    def resolve(self, name: str) -> list[str]:
        """Resolve a name to all matching handles with fuzzy matching.

        Performs case-insensitive matching, including prefix and substring matching
        on first and last names.

        Args:
            name: The name to look up (e.g., "John" or "John Doe").

        Returns:
            List of all handles (phone numbers, emails) for matching contacts.
        """
        if not name:
            return []

        name_lower = name.lower().strip()
        matches = []

        # Exact case-insensitive match
        for canonical_name, handles in self._name_to_handles.items():
            if canonical_name.lower() == name_lower:
                matches.extend(handles)
                return sorted(matches)

        # Fuzzy matching: first name, last name, or substring
        name_parts = name_lower.split()

        for canonical_name, handles in self._name_to_handles.items():
            canonical_lower = canonical_name.lower()
            canonical_parts = canonical_lower.split()

            # Check if any input part matches any canonical part (prefix or substring)
            for input_part in name_parts:
                for canonical_part in canonical_parts:
                    if canonical_part.startswith(input_part) or input_part in canonical_part:
                        matches.extend(handles)
                        break

        return sorted(matches)

    def lookup(self, handle: str) -> str | None:
        """Look up the display name for a given handle.

        Args:
            handle: The handle to look up (phone number, email, or iMessage handle).

        Returns:
            The contact's display name, or None if not found.
        """
        normalized = self.normalize_handle(handle)
        return self._handle_to_name.get(normalized)

    def normalize_handle(self, handle: str) -> str:
        """Normalize a handle for consistent comparison.

        Normalizes phone numbers by stripping formatting characters and
        lowercase email addresses.

        Args:
            handle: The handle to normalize.

        Returns:
            The normalized handle.
        """
        if not handle:
            return ""

        handle = handle.strip()

        if not handle:
            return ""

        # For phone numbers: strip formatting characters
        # Keep only digits and leading +
        if handle.startswith("+") or handle[0].isdigit():
            # Remove all non-alphanumeric except +
            normalized = "".join(c for c in handle if c.isalnum() or c == "+")
            return normalized

        # For emails and other handles: lowercase
        return handle.lower()

    def count(self) -> int:
        """Return the total number of unique contacts in the index.

        Returns:
            Number of unique contact names.
        """
        return len(self._name_to_handles)
