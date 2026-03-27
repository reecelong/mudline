"""SafariExtractor — extract Safari browsing history and bookmarks.

Parses HomeDomain/Library/Safari/History.db and Bookmarks.db to extract
browsing history with visit counts and bookmarks with folder organization.
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


def _cocoa_seconds_to_datetime(cocoa_seconds: float) -> datetime:
    """Convert Cocoa epoch seconds to datetime.

    Args:
        cocoa_seconds: Seconds since 2001-01-01.

    Returns:
        Converted datetime.
    """
    return COCOA_EPOCH + timedelta(seconds=cocoa_seconds)


class SafariExtractor:
    """Extract Safari history and bookmarks."""

    @property
    def domain(self) -> str:
        """iOS backup domain for Safari."""
        return "HomeDomain"

    @property
    def data_type(self) -> str:
        """Document type produced by this extractor."""
        return DocumentType.SAFARI.value

    def can_extract(self, resolver: ManifestResolver) -> bool:
        """Check if Safari data exists in the backup.

        Args:
            resolver: ManifestResolver for the target backup.

        Returns:
            True if History.db exists, False otherwise.
        """
        return resolver.file_exists(
            self.domain, "Library/Safari/History.db"
        )

    def extract(self, resolver: ManifestResolver) -> Iterator[Document]:
        """Extract Safari history and bookmarks.

        Args:
            resolver: ManifestResolver for the target backup.

        Yields:
            Document objects for each history entry and bookmark.

        Raises:
            FileNotFoundError: If History.db is missing.
            ExtractionError: If the database schema is unexpected.
        """
        backup_id = self._build_backup_id(resolver)
        backup_timestamp = self._get_backup_timestamp(resolver)

        # Extract history
        yield from self._extract_history(resolver, backup_id, backup_timestamp)

        # Extract bookmarks (if available)
        try:
            yield from self._extract_bookmarks(resolver, backup_id, backup_timestamp)
        except FileNotFoundError:
            logger.debug("Bookmarks.db not found, skipping bookmarks extraction")

    def _extract_history(
        self,
        resolver: ManifestResolver,
        backup_id: str,
        backup_timestamp: datetime,
    ) -> Iterator[Document]:
        """Extract Safari browsing history.

        Args:
            resolver: ManifestResolver for the target backup.
            backup_id: Backup identifier.
            backup_timestamp: Backup timestamp.

        Yields:
            Document objects for each history entry.

        Raises:
            FileNotFoundError: If History.db is missing.
            ExtractionError: If the database is corrupt.
        """
        try:
            history_path = resolver.resolve(
                self.domain, "Library/Safari/History.db"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Safari History database not found in backup: "
                f"{self.domain}/Library/Safari/History.db"
            ) from e

        try:
            conn = sqlite3.connect(
                f"file:{history_path}?mode=ro&immutable=1", uri=True
            )
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise ExtractionError(f"Failed to open History database: {history_path}") from e

        try:
            # Load history items
            history_items = self._load_history_items(conn)

            # Fetch all visits
            cursor = conn.execute(
                """
                SELECT
                    id, history_item, visit_time, title
                FROM history_visits
                ORDER BY visit_time ASC
                """
            )

            for row in cursor:
                visit_id = row["id"]
                history_item_id = row["history_item"]
                visit_time = row["visit_time"]
                title = row["title"] or ""

                # Get the URL and visit count from history_items
                if history_item_id not in history_items:
                    continue

                url, visit_count = history_items[history_item_id]

                # Convert timestamp
                timestamp = None
                if visit_time is not None:
                    try:
                        timestamp = _cocoa_seconds_to_datetime(visit_time)
                    except (ValueError, OverflowError):
                        logger.warning(
                            "Invalid timestamp for visit %d: %s",
                            visit_id,
                            visit_time,
                        )

                # Build text content
                text = title or url or "Safari visit"

                # Build metadata
                metadata = {
                    "url": url,
                    "visit_count": visit_count,
                }

                # Create source
                source = Source(
                    backup_id=backup_id,
                    domain=self.domain,
                    relative_path=f"Library/Safari/History.db/visit/{visit_id}",
                    backup_timestamp=backup_timestamp,
                )

                # Create document
                doc = Document(
                    type=DocumentType.SAFARI,
                    text=text,
                    timestamp=timestamp,
                    metadata=metadata,
                    source=source,
                )

                yield doc

        except sqlite3.Error as e:
            raise ExtractionError(f"Database error while extracting history: {e}") from e
        finally:
            conn.close()

    def _extract_bookmarks(
        self,
        resolver: ManifestResolver,
        backup_id: str,
        backup_timestamp: datetime,
    ) -> Iterator[Document]:
        """Extract Safari bookmarks.

        Args:
            resolver: ManifestResolver for the target backup.
            backup_id: Backup identifier.
            backup_timestamp: Backup timestamp.

        Yields:
            Document objects for each bookmark.

        Raises:
            FileNotFoundError: If Bookmarks.db is missing.
            ExtractionError: If the database is corrupt.
        """
        try:
            bookmarks_path = resolver.resolve(
                self.domain, "Library/Safari/Bookmarks.db"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Safari Bookmarks database not found in backup: "
                f"{self.domain}/Library/Safari/Bookmarks.db"
            ) from e

        try:
            conn = sqlite3.connect(
                f"file:{bookmarks_path}?mode=ro&immutable=1", uri=True
            )
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise ExtractionError(f"Failed to open Bookmarks database: {bookmarks_path}") from e

        try:
            # Load folder mappings
            folders = self._load_bookmark_folders(conn)

            # Fetch all bookmarks
            cursor = conn.execute(
                """
                SELECT
                    id, title, url, parent
                FROM bookmarks
                WHERE url IS NOT NULL
                ORDER BY id ASC
                """
            )

            for row in cursor:
                bookmark_id = row["id"]
                title = row["title"] or ""
                url = row["url"] or ""
                parent_id = row["parent"]

                # Skip entries without URLs
                if not url:
                    continue

                # Get folder name
                folder = None
                if parent_id is not None:
                    folder = folders.get(parent_id)

                # Build text content
                text = title or url or "Bookmark"

                # Build metadata
                metadata = {
                    "url": url,
                    "folder": folder,
                }

                # Create source
                source = Source(
                    backup_id=backup_id,
                    domain=self.domain,
                    relative_path=f"Library/Safari/Bookmarks.db/bookmark/{bookmark_id}",
                    backup_timestamp=backup_timestamp,
                )

                # Create document
                doc = Document(
                    type=DocumentType.SAFARI,
                    text=text,
                    source=source,
                    metadata=metadata,
                )

                yield doc

        except sqlite3.Error as e:
            raise ExtractionError(f"Database error while extracting bookmarks: {e}") from e
        finally:
            conn.close()

    def _load_history_items(self, conn: sqlite3.Connection) -> dict[int, tuple[str, int]]:
        """Load history items (ID → (URL, visit_count)).

        Args:
            conn: SQLite connection to History.db.

        Returns:
            Mapping of history_item ID to (URL, visit_count).
        """
        items: dict[int, tuple[str, int]] = {}
        try:
            cursor = conn.execute(
                "SELECT id, url, visit_count FROM history_items WHERE url IS NOT NULL"
            )
            for row in cursor:
                items[row[0]] = (row[1], row[2])
        except sqlite3.Error as e:
            logger.warning("Failed to load history items: %s", e)
        return items

    def _load_bookmark_folders(self, conn: sqlite3.Connection) -> dict[int, str]:
        """Load bookmark folder mappings (folder_id → folder_title).

        Args:
            conn: SQLite connection to Bookmarks.db.

        Returns:
            Mapping of folder ID to folder title.
        """
        folders: dict[int, str] = {}
        try:
            cursor = conn.execute(
                "SELECT id, title FROM bookmarks WHERE title IS NOT NULL AND url IS NULL"
            )
            for row in cursor:
                folders[row[0]] = row[1]
        except sqlite3.Error as e:
            logger.warning("Failed to load bookmark folders: %s", e)
        return folders

    def _build_backup_id(self, resolver: ManifestResolver) -> str:
        """Build a backup ID from backup path.

        Args:
            resolver: ManifestResolver.

        Returns:
            A string identifier for the backup.
        """
        return resolver.backup_path.name

    def _get_backup_timestamp(self, resolver: ManifestResolver) -> datetime:
        """Get the backup timestamp from Info.plist.

        Args:
            resolver: ManifestResolver.

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
