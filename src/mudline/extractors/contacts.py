"""ContactExtractor — extract contacts from iOS AddressBook.

Parses HomeDomain/Library/AddressBook/AddressBook.sqlitedb to extract
contact information including phone numbers, emails, and organizations.
"""

from __future__ import annotations

import logging
import plistlib
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from mudline.exceptions import ExtractionError
from mudline.models.document import Document, DocumentType, Source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mudline.foundation.manifest import ManifestResolver

logger = logging.getLogger(__name__)

# Property type constants from ABMultiValue table
PROPERTY_PHONE = 3
PROPERTY_EMAIL = 4


class ContactExtractor:
    """Extract contacts from iOS AddressBook.sqlitedb.

    Parses the AddressBook database to extract all contacts with their
    associated phone numbers, emails, and organization information.

    Attributes:
        domain: The iOS backup domain ("HomeDomain").
        data_type: The document type ("contact").
    """

    @property
    def domain(self) -> str:
        """Return the iOS backup domain."""
        return "HomeDomain"

    @property
    def data_type(self) -> str:
        """Return the document type."""
        return "contact"

    def can_extract(self, resolver: ManifestResolver) -> bool:
        """Check if the AddressBook database exists in the backup.

        Args:
            resolver: ManifestResolver for the target backup.

        Returns:
            True if AddressBook.sqlitedb exists, False otherwise.
        """
        return resolver.file_exists(self.domain, "Library/AddressBook/AddressBook.sqlitedb")

    def extract(self, resolver: ManifestResolver) -> Iterator[Document]:
        """Extract all contacts from AddressBook.sqlitedb.

        Args:
            resolver: ManifestResolver for the target backup.

        Yields:
            Document objects for each contact found.

        Raises:
            FileNotFoundError: If AddressBook.sqlitedb is not in the backup.
            ExtractionError: If the database schema is unexpected or data is corrupt.
        """
        contacts_path = resolver.resolve(
            self.domain, "Library/AddressBook/AddressBook.sqlitedb"
        )

        try:
            conn = sqlite3.connect(
                f"file:{contacts_path}?mode=ro&immutable=1", uri=True
            )
            conn.row_factory = sqlite3.Row

            # Fetch all persons and their multi-value records
            cursor = conn.execute(
                """
                SELECT
                    p.ROWID,
                    p.First,
                    p.Last,
                    p.Organization
                FROM ABPerson p
                ORDER BY p.ROWID
                """
            )
            persons = cursor.fetchall()

            # Build a map of person ROWID → their phone/email values
            persons_map: dict[int, dict] = {}
            for person in persons:
                rowid = person["ROWID"]
                persons_map[rowid] = {
                    "first": person["First"],
                    "last": person["Last"],
                    "organization": person["Organization"],
                    "phones": [],
                    "emails": [],
                }

            # Fetch all multi-value records (phones and emails)
            cursor = conn.execute(
                """
                SELECT record_id, property, value
                FROM ABMultiValue
                ORDER BY record_id, property, UID
                """
            )
            for row in cursor.fetchall():
                record_id = row["record_id"]
                prop = row["property"]
                value = row["value"]

                if record_id not in persons_map:
                    continue

                if prop == PROPERTY_PHONE:
                    persons_map[record_id]["phones"].append(value)
                elif prop == PROPERTY_EMAIL:
                    persons_map[record_id]["emails"].append(value)

            conn.close()

        except sqlite3.Error as e:
            raise ExtractionError(
                f"Failed to parse AddressBook.sqlitedb: {e}"
            ) from e

        # Get backup metadata for provenance
        backup_id = f"{resolver.backup_path.name}"
        backup_timestamp = self._get_backup_timestamp(resolver)

        # Generate Document for each contact
        for _rowid, contact_data in persons_map.items():
            first = contact_data["first"] or ""
            last = contact_data["last"] or ""
            org = contact_data["organization"]
            phones = contact_data["phones"]
            emails = contact_data["emails"]

            # Build display name
            display_name = f"{first} {last}".strip()
            if not display_name:
                display_name = "Unknown Contact"

            # Build text content: "FirstName LastName — org — phone — email"
            text_parts = [display_name]
            if org:
                text_parts.append(org)
            if phones:
                text_parts.append(" ".join(phones))
            if emails:
                text_parts.append(" ".join(emails))

            text = " — ".join(text_parts)

            # Build handles (all phone + email)
            handles = phones + emails

            # Build metadata
            metadata = {
                "phones": phones,
                "emails": emails,
                "organization": org,
                "handles": handles,
            }

            # Create document
            source = Source(
                backup_id=backup_id,
                domain=self.domain,
                relative_path="Library/AddressBook/AddressBook.sqlitedb",
                backup_timestamp=backup_timestamp,
            )

            doc = Document(
                type=DocumentType.CONTACT,
                text=text,
                source=source,
                timestamp=None,  # Contacts have no timestamp
                metadata=metadata,
            )

            yield doc

    def get_handle_map(self, resolver: ManifestResolver) -> dict[str, str]:
        """Build a map of each phone/email → contact display name.

        Args:
            resolver: ManifestResolver for the target backup.

        Returns:
            Dictionary mapping each phone/email handle to "First Last" display name.

        Raises:
            FileNotFoundError: If AddressBook.sqlitedb is not in the backup.
            ExtractionError: If the database schema is unexpected or data is corrupt.
        """
        handle_map: dict[str, str] = {}

        contacts_path = resolver.resolve(
            self.domain, "Library/AddressBook/AddressBook.sqlitedb"
        )

        try:
            conn = sqlite3.connect(
                f"file:{contacts_path}?mode=ro&immutable=1", uri=True
            )
            conn.row_factory = sqlite3.Row

            # Fetch all persons
            cursor = conn.execute(
                """
                SELECT
                    p.ROWID,
                    p.First,
                    p.Last
                FROM ABPerson p
                """
            )
            persons = {
                row["ROWID"]: (row["First"], row["Last"]) for row in cursor.fetchall()
            }

            # Fetch all multi-value records
            cursor = conn.execute(
                """
                SELECT record_id, property, value
                FROM ABMultiValue
                WHERE property IN (?, ?)
                """,
                (PROPERTY_PHONE, PROPERTY_EMAIL),
            )

            for row in cursor.fetchall():
                record_id = row["record_id"]
                value = row["value"]

                if record_id not in persons:
                    continue

                first, last = persons[record_id]
                display_name = f"{first or ''} {last or ''}".strip()
                if not display_name:
                    display_name = "Unknown Contact"

                handle_map[value] = display_name

            conn.close()

        except sqlite3.Error as e:
            raise ExtractionError(
                f"Failed to parse AddressBook.sqlitedb for handle map: {e}"
            ) from e

        return handle_map

    def _get_backup_timestamp(self, resolver: ManifestResolver) -> datetime:
        """Extract backup timestamp from Info.plist.

        Args:
            resolver: ManifestResolver for the target backup.

        Returns:
            The backup creation timestamp.
        """
        info_plist = resolver.backup_path / "Info.plist"

        try:
            with open(info_plist, "rb") as f:
                info = plistlib.load(f)
                backup_date = info.get("Last Backup Date")
                if isinstance(backup_date, datetime):
                    return backup_date
        except Exception as e:
            logger.warning("Could not read backup timestamp from Info.plist: %s", e)

        # Fallback to epoch if Info.plist is missing or malformed
        return datetime.now()
