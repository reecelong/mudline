"""CalendarExtractor — extract calendar events from iOS Calendar app.

Parses HomeDomain/Library/Calendar/Calendar.sqlitedb to extract events with
location, attendees, recurrence, and timestamps.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from mudline.exceptions import ExtractionError
from mudline.models.document import Document, DocumentType, Source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mudline.models import ResourceResolver

logger = logging.getLogger(__name__)

COCOA_EPOCH = datetime(2001, 1, 1)


def _cocoa_seconds_to_datetime(cocoa_seconds: float) -> datetime:
    """Convert Cocoa epoch seconds to datetime.

    Args:
        cocoa_seconds: Seconds since 2001-01-01.

    Returns:
        Converted datetime.
    """
    return COCOA_EPOCH + timedelta(seconds=cocoa_seconds)


class CalendarExtractor:
    """Extract calendar events from iOS Calendar app."""

    @property
    def domain(self) -> str:
        """iOS backup domain for calendar."""
        return "HomeDomain"

    @property
    def data_type(self) -> str:
        """Document type produced by this extractor."""
        return DocumentType.CALENDAR.value

    def can_extract(self, resolver: ResourceResolver) -> bool:
        """Check if the Calendar database exists in the backup.

        Args:
            resolver: ResourceResolver for the target backup.

        Returns:
            True if Calendar.sqlitedb exists, False otherwise.
        """
        return resolver.file_exists(self.domain, "Library/Calendar/Calendar.sqlitedb")

    def extract(self, resolver: ResourceResolver) -> Iterator[Document]:
        """Extract all calendar events from the Calendar database.

        Args:
            resolver: ResourceResolver for the target backup.

        Yields:
            Document objects for each calendar event.

        Raises:
            FileNotFoundError: If Calendar.sqlitedb is missing.
            ExtractionError: If the database schema is unexpected.
        """
        try:
            calendar_path = resolver.resolve(self.domain, "Library/Calendar/Calendar.sqlitedb")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Calendar database not found in backup: "
                f"{self.domain}/Library/Calendar/Calendar.sqlitedb"
            ) from e

        try:
            conn = sqlite3.connect(f"file:{calendar_path}?mode=ro&immutable=1", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise ExtractionError(f"Failed to open Calendar database: {calendar_path}") from e

        try:
            # Load attendee mappings
            attendees_map = self._load_attendees(conn)

            # Load recurrence mappings
            recurrence_map = self._load_recurrence(conn)

            # Fetch all calendar items
            cursor = conn.execute(
                """
                SELECT
                    ROWID, summary, description, start_date, end_date,
                    all_day, calendar_id
                FROM CalendarItem
                ORDER BY start_date ASC
                """
            )

            backup_id = self._build_backup_id(resolver)
            backup_timestamp = self._get_backup_timestamp(resolver)

            for row in cursor:
                item_id = row["ROWID"]
                summary = row["summary"] or ""
                description = row["description"] or ""
                start_date = row["start_date"]
                end_date = row["end_date"]
                all_day = bool(row["all_day"])

                # Convert timestamps
                timestamp = None
                end_timestamp = None
                if start_date is not None:
                    try:
                        timestamp = _cocoa_seconds_to_datetime(start_date)
                    except (ValueError, OverflowError):
                        logger.warning(
                            "Invalid start timestamp for event %d: %s",
                            item_id,
                            start_date,
                        )

                if end_date is not None:
                    try:
                        end_timestamp = _cocoa_seconds_to_datetime(end_date)
                    except (ValueError, OverflowError):
                        logger.warning(
                            "Invalid end timestamp for event %d: %s",
                            item_id,
                            end_date,
                        )

                # Get attendees
                attendees = attendees_map.get(item_id, [])

                # Get recurrence
                recurrence = recurrence_map.get(item_id)

                # Build text content: "Title — location — notes"
                text_parts = [summary]
                if description:
                    text_parts.append(description)
                text = " — ".join(filter(None, text_parts))

                # Extract location from description if not available as separate field
                location = None
                if description and "Location:" in description:
                    # Simple extraction: look for "Location: ..." in description
                    lines = description.split("\n")
                    for line in lines:
                        if line.startswith("Location:"):
                            location = line.replace("Location:", "").strip()
                            break

                # Build metadata (all values must be JSON-serializable)
                metadata = {
                    "location": location,
                    "attendees": attendees,
                    "is_all_day": all_day,
                    "end_time": end_timestamp.isoformat() if end_timestamp else None,
                    "recurrence": recurrence,
                }

                # Create source
                source = Source(
                    backup_id=backup_id,
                    domain=self.domain,
                    relative_path=f"Library/Calendar/Calendar.sqlitedb/event/{item_id}",
                    backup_timestamp=backup_timestamp,
                )

                # Create document
                doc = Document(
                    type=DocumentType.CALENDAR,
                    text=text,
                    timestamp=timestamp,
                    metadata=metadata,
                    source=source,
                )

                yield doc

        except sqlite3.Error as e:
            raise ExtractionError(f"Database error while extracting calendar: {e}") from e
        finally:
            conn.close()

    def _load_attendees(self, conn: sqlite3.Connection) -> dict[int, list[str]]:
        """Load attendee mappings for calendar items.

        Handles schema variations across iOS versions by probing for the
        correct table and column names.

        Args:
            conn: SQLite connection to Calendar.sqlitedb.

        Returns:
            Mapping of item ROWID to list of attendee email addresses.
        """
        attendees_map: dict[int, list[str]] = {}

        # Probe for known table/column combinations across iOS versions
        queries = [
            "SELECT item_id, address FROM Attendee WHERE address IS NOT NULL",
            "SELECT owner_id, address FROM Attendee WHERE address IS NOT NULL",
            "SELECT item_id, ZADDRESS FROM ZATTENDEE WHERE ZADDRESS IS NOT NULL",
            "SELECT ZITEM, ZADDRESS FROM ZATTENDEE WHERE ZADDRESS IS NOT NULL",
        ]

        for query in queries:
            try:
                cursor = conn.execute(query)
                for row in cursor:
                    item_id = row[0]
                    address = row[1]
                    if item_id not in attendees_map:
                        attendees_map[item_id] = []
                    attendees_map[item_id].append(address)
                return attendees_map
            except sqlite3.OperationalError:
                continue

        logger.debug("No attendee table found in Calendar database")
        return attendees_map

    def _load_recurrence(self, conn: sqlite3.Connection) -> dict[int, str]:
        """Load recurrence information for calendar items.

        Handles schema variations across iOS versions.

        Args:
            conn: SQLite connection to Calendar.sqlitedb.

        Returns:
            Mapping of item ROWID to recurrence frequency string.
        """
        recurrence_map: dict[int, str] = {}

        queries = [
            "SELECT item_id, frequency FROM Recurrence WHERE frequency IS NOT NULL",
            "SELECT owner_id, frequency FROM Recurrence WHERE frequency IS NOT NULL",
            "SELECT ZITEM, ZFREQUENCY FROM ZRECURRENCERULE WHERE ZFREQUENCY IS NOT NULL",
        ]

        for query in queries:
            try:
                cursor = conn.execute(query)
                for row in cursor:
                    item_id = row[0]
                    frequency = row[1]
                    recurrence_map[item_id] = str(frequency)
                return recurrence_map
            except sqlite3.OperationalError:
                continue

        logger.debug("No recurrence table found in Calendar database")
        return recurrence_map

    def _build_backup_id(self, resolver: ResourceResolver) -> str:
        """Build a backup ID from backup path.

        Args:
            resolver: ResourceResolver.

        Returns:
            A string identifier for the backup.
        """
        return resolver.backup_path.name

    def _get_backup_timestamp(self, resolver: ResourceResolver) -> datetime:
        """Get the backup timestamp from Info.plist.

        Args:
            resolver: ResourceResolver.

        Returns:
            Backup timestamp, or current time if unavailable.
        """
        import plistlib

        info_plist = resolver.backup_path / "Info.plist"
        if info_plist.exists():
            try:
                with open(info_plist, "rb") as f:
                    info = plistlib.load(f)
                    if "Last Backup Date" in info:
                        return info["Last Backup Date"]
            except Exception as e:
                logger.warning("Failed to read Info.plist: %s", e)

        return datetime.now()
