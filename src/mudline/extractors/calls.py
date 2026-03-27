"""CallHistoryExtractor — extract call history from iOS backups.

Parses HomeDomain/Library/CallHistoryDB/CallHistory.storedata to extract calls with:
- Phone number / contact handle
- Duration in seconds
- Timestamp (Cocoa epoch seconds → datetime)
- Call type (incoming, outgoing, missed)

iOS call history timestamps are stored as seconds since Cocoa epoch (2001-01-01),
NOT nanoseconds like SMS messages.
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

    from mudline.foundation.manifest import ManifestResolver

logger = logging.getLogger(__name__)

COCOA_EPOCH = datetime(2001, 1, 1)


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string (e.g., "3m 5s" or "42s").
    """
    total_seconds = int(seconds)
    if total_seconds == 0:
        return "0s"
    minutes = total_seconds // 60
    secs = total_seconds % 60
    if minutes > 0 and secs > 0:
        return f"{minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return f"{secs}s"


def _cocoa_seconds_to_datetime(cocoa_seconds: float) -> datetime:
    """Convert Cocoa epoch seconds to datetime.

    Args:
        cocoa_seconds: Seconds since 2001-01-01.

    Returns:
        Converted datetime.
    """
    return COCOA_EPOCH + timedelta(seconds=cocoa_seconds)


class CallHistoryExtractor:
    """Extract call history from iOS backup."""

    @property
    def domain(self) -> str:
        """iOS backup domain for call history."""
        return "HomeDomain"

    @property
    def data_type(self) -> str:
        """Document type produced by this extractor."""
        return DocumentType.CALL.value

    def can_extract(self, resolver: ManifestResolver) -> bool:
        """Check if the call history database exists in the backup.

        Args:
            resolver: ManifestResolver for the target backup.

        Returns:
            True if CallHistory.storedata exists, False otherwise.
        """
        return resolver.file_exists(
            self.domain, "Library/CallHistoryDB/CallHistory.storedata"
        )

    def extract(self, resolver: ManifestResolver) -> Iterator[Document]:
        """Extract all calls from the call history database.

        Args:
            resolver: ManifestResolver for the target backup.

        Yields:
            Document objects for each call.

        Raises:
            FileNotFoundError: If CallHistory.storedata is missing.
            ExtractionError: If the database schema is unexpected.
        """
        try:
            call_history_path = resolver.resolve(
                self.domain, "Library/CallHistoryDB/CallHistory.storedata"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Call history database not found in backup: "
                f"{self.domain}/Library/CallHistoryDB/CallHistory.storedata"
            ) from e

        # Connect to the call history database in read-only mode
        try:
            conn = sqlite3.connect(
                f"file:{call_history_path}?mode=ro&immutable=1", uri=True
            )
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise ExtractionError(
                f"Failed to open call history database: {call_history_path}"
            ) from e

        try:
            # Fetch all call records
            cursor = conn.execute(
                """
                SELECT
                    Z_PK,
                    ZADDRESS,
                    ZDURATION,
                    ZDATE,
                    ZORIGINATED,
                    ZANSWERED
                FROM ZCALLRECORD
                ORDER BY ZDATE ASC
                """
            )

            backup_id = self._build_backup_id(resolver)
            backup_timestamp = self._get_backup_timestamp(resolver)

            for row in cursor:
                call_id = row["Z_PK"]
                address = row["ZADDRESS"] or ""
                duration = row["ZDURATION"] or 0.0
                date_seconds = row["ZDATE"]
                originated = bool(row["ZORIGINATED"])
                answered = bool(row["ZANSWERED"])

                # Convert timestamp
                try:
                    timestamp = _cocoa_seconds_to_datetime(date_seconds)
                except (ValueError, OverflowError):
                    logger.warning(
                        "Invalid timestamp for call %d: %s, skipping",
                        call_id,
                        date_seconds,
                    )
                    continue

                # Determine call type
                if originated:
                    call_type = "outgoing"
                elif not answered:
                    call_type = "missed"
                else:
                    call_type = "incoming"

                # Format duration
                duration_str = _format_duration(duration)

                # Build text content
                text = f"Call with {address}, {duration_str}, {call_type}"

                # Build metadata
                metadata = {
                    "handle": address,
                    "duration_seconds": int(duration),
                    "call_type": call_type,
                }

                # Create source
                source = Source(
                    backup_id=backup_id,
                    domain=self.domain,
                    relative_path=f"Library/CallHistoryDB/CallHistory.storedata/call/{call_id}",
                    backup_timestamp=backup_timestamp,
                )

                # Create document
                doc = Document(
                    type=DocumentType.CALL,
                    text=text,
                    timestamp=timestamp,
                    metadata=metadata,
                    source=source,
                )

                yield doc

        except sqlite3.Error as e:
            raise ExtractionError(f"Database error while extracting calls: {e}") from e
        finally:
            conn.close()

    def _build_backup_id(self, resolver: ManifestResolver) -> str:
        """Build a backup ID from backup path.

        Args:
            resolver: ManifestResolver.

        Returns:
            A string identifier for the backup.
        """
        # Use the backup directory name as the backup_id
        backup_name = resolver.backup_path.name
        return backup_name

    def _get_backup_timestamp(self, resolver: ManifestResolver) -> datetime:
        """Get the backup timestamp from Info.plist or Status.plist.

        Args:
            resolver: ManifestResolver.

        Returns:
            Backup timestamp, or current time if unavailable.
        """
        import plistlib

        # Try Info.plist first
        info_plist = resolver.backup_path / "Info.plist"
        if info_plist.exists():
            try:
                with open(info_plist, "rb") as f:
                    info = plistlib.load(f)
                    if "Last Backup Date" in info:
                        return info["Last Backup Date"]
            except Exception as e:
                logger.warning("Failed to read Info.plist: %s", e)

        # Fallback to Status.plist
        status_plist = resolver.backup_path / "Status.plist"
        if status_plist.exists():
            try:
                with open(status_plist, "rb") as f:
                    status = plistlib.load(f)
                    if "Date" in status:
                        return status["Date"]
            except Exception as e:
                logger.warning("Failed to read Status.plist: %s", e)

        # Fallback to current time
        return datetime.now()
