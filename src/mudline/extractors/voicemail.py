"""VoicemailExtractor — extract voicemail messages from iOS.

Parses HomeDomain/Library/Voicemail/voicemail.db to extract voicemail
metadata including sender, duration, transcription, and timestamps.
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


class VoicemailExtractor:
    """Extract voicemail messages from iOS."""

    @property
    def domain(self) -> str:
        """iOS backup domain for voicemail."""
        return "HomeDomain"

    @property
    def data_type(self) -> str:
        """Document type produced by this extractor."""
        return DocumentType.VOICEMAIL.value

    def can_extract(self, resolver: ResourceResolver) -> bool:
        """Check if the Voicemail database exists in the backup.

        Args:
            resolver: ResourceResolver for the target backup.

        Returns:
            True if voicemail.db exists, False otherwise.
        """
        return resolver.file_exists(
            self.domain, "Library/Voicemail/voicemail.db"
        )

    def extract(self, resolver: ResourceResolver) -> Iterator[Document]:
        """Extract all voicemail messages from the Voicemail database.

        Args:
            resolver: ResourceResolver for the target backup.

        Yields:
            Document objects for each voicemail.

        Raises:
            FileNotFoundError: If voicemail.db is missing.
            ExtractionError: If the database schema is unexpected.
        """
        try:
            voicemail_path = resolver.resolve(
                self.domain, "Library/Voicemail/voicemail.db"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Voicemail database not found in backup: "
                f"{self.domain}/Library/Voicemail/voicemail.db"
            ) from e

        try:
            conn = sqlite3.connect(
                f"file:{voicemail_path}?mode=ro&immutable=1", uri=True
            )
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise ExtractionError(f"Failed to open Voicemail database: {voicemail_path}") from e

        try:
            # Fetch all voicemails
            cursor = conn.execute(
                """
                SELECT
                    ROWID, sender, date, duration, trashed_date,
                    transcription, receiver
                FROM voicemail
                ORDER BY date ASC
                """
            )

            backup_id = self._build_backup_id(resolver)
            backup_timestamp = self._get_backup_timestamp(resolver)

            for row in cursor:
                voicemail_id = row["ROWID"]
                sender = row["sender"] or ""
                date_value = row["date"]
                duration = row["duration"]
                trashed_date = row["trashed_date"]
                transcription = row["transcription"]
                receiver = row["receiver"]

                # Skip voicemails that have been deleted (trashed_date is set)
                if trashed_date is not None:
                    continue

                # Convert timestamp
                timestamp = None
                if date_value is not None:
                    try:
                        timestamp = _cocoa_seconds_to_datetime(date_value)
                    except (ValueError, OverflowError):
                        logger.warning(
                            "Invalid timestamp for voicemail %d: %s",
                            voicemail_id,
                            date_value,
                        )

                # Build duration in seconds
                duration_seconds = int(duration) if duration else 0

                # Build text content
                text = transcription or (
                    f"Voicemail from {sender}" if sender else "Voicemail"
                )

                # Build metadata
                metadata = {
                    "handle": sender or receiver or "Unknown",
                    "duration_seconds": duration_seconds,
                    "transcription": transcription,
                }

                # Create source
                source = Source(
                    backup_id=backup_id,
                    domain=self.domain,
                    relative_path=f"Library/Voicemail/voicemail.db/voicemail/{voicemail_id}",
                    backup_timestamp=backup_timestamp,
                )

                # Create document
                doc = Document(
                    type=DocumentType.VOICEMAIL,
                    text=text,
                    timestamp=timestamp,
                    metadata=metadata,
                    source=source,
                )

                yield doc

        except sqlite3.Error as e:
            raise ExtractionError(f"Database error while extracting voicemail: {e}") from e
        finally:
            conn.close()

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
