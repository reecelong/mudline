"""NoteExtractor — extract notes from iOS Notes app.

Parses HomeDomain/Library/Notes/NoteStore.sqlite to extract note metadata
including title, content snippet, folder, and timestamps.

Note: Full protobuf decoding of ZDATA for rich text is left as TODO.
For now, we use ZSNIPPET as the text content.
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


class NoteExtractor:
    """Extract notes from iOS Notes app."""

    @property
    def domain(self) -> str:
        """iOS backup domain for notes."""
        return "HomeDomain"

    @property
    def data_type(self) -> str:
        """Document type produced by this extractor."""
        return DocumentType.NOTE.value

    def can_extract(self, resolver: ResourceResolver) -> bool:
        """Check if the Notes database exists in the backup.

        Args:
            resolver: ResourceResolver for the target backup.

        Returns:
            True if NoteStore.sqlite exists, False otherwise.
        """
        return resolver.file_exists(self.domain, "Library/Notes/NoteStore.sqlite")

    def extract(self, resolver: ResourceResolver) -> Iterator[Document]:
        """Extract all notes from the Notes database.

        Args:
            resolver: ResourceResolver for the target backup.

        Yields:
            Document objects for each note.

        Raises:
            FileNotFoundError: If NoteStore.sqlite is missing.
            ExtractionError: If the database schema is unexpected.
        """
        try:
            notes_path = resolver.resolve(self.domain, "Library/Notes/NoteStore.sqlite")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Notes database not found in backup: {self.domain}/Library/Notes/NoteStore.sqlite"
            ) from e

        try:
            conn = sqlite3.connect(f"file:{notes_path}?mode=ro&immutable=1", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise ExtractionError(f"Failed to open Notes database: {notes_path}") from e

        try:
            # Load folder mapping (folder_id → folder_title)
            folders = self._load_folders(conn)

            # Fetch all notes
            # Filter to actual notes (typically have ZSNIPPET or ZDATA)
            # Skip folders and other organizational items
            cursor = conn.execute(
                """
                SELECT
                    Z_PK, ZTITLE, ZSNIPPET, ZMODIFICATIONDATE,
                    ZCREATIONDATE, ZDATA, ZFOLDER, ZACCOUNT
                FROM ZICCLOUDSYNCINGOBJECT
                WHERE (ZSNIPPET IS NOT NULL AND ZSNIPPET != '')
                   OR (ZDATA IS NOT NULL AND ZDATA != '')
                   OR (ZTITLE IS NOT NULL AND ZTITLE != '' AND ZFOLDER IS NOT NULL)
                ORDER BY ZCREATIONDATE ASC
                """
            )

            backup_id = self._build_backup_id(resolver)
            backup_timestamp = self._get_backup_timestamp(resolver)

            for row in cursor:
                note_id = row["Z_PK"]
                title = row["ZTITLE"] or ""
                snippet = row["ZSNIPPET"] or ""
                modification_date = row["ZMODIFICATIONDATE"]
                creation_date = row["ZCREATIONDATE"]
                zdata = row["ZDATA"]
                folder_id = row["ZFOLDER"]

                # Convert timestamp (use creation date if available, else modification)
                timestamp = None
                ts_value = creation_date or modification_date
                if ts_value is not None:
                    try:
                        timestamp = _cocoa_seconds_to_datetime(ts_value)
                    except (ValueError, OverflowError):
                        logger.warning(
                            "Invalid timestamp for note %d: %s",
                            note_id,
                            ts_value,
                        )

                # Determine text content
                # TODO: Decode ZDATA protobuf for full rich text content
                text = snippet or title or ""

                # Get folder name
                folder = None
                if folder_id is not None:
                    folder = folders.get(folder_id)

                # Check if note has attachments (ZDATA being non-null may indicate content)
                has_attachments = zdata is not None and len(zdata) > 0

                # Build metadata
                metadata = {
                    "folder": folder,
                    "has_attachments": has_attachments,
                }

                # Build title (prefer explicit title, fallback to first line of snippet)
                display_title = title
                if not display_title and snippet:
                    # Use first line of snippet as title
                    display_title = snippet.split("\n")[0][:100]

                # Create source
                source = Source(
                    backup_id=backup_id,
                    domain=self.domain,
                    relative_path=f"Library/Notes/NoteStore.sqlite/note/{note_id}",
                    backup_timestamp=backup_timestamp,
                )

                # Create document
                doc = Document(
                    type=DocumentType.NOTE,
                    text=text,
                    timestamp=timestamp,
                    metadata=metadata,
                    source=source,
                )

                yield doc

        except sqlite3.Error as e:
            raise ExtractionError(f"Database error while extracting notes: {e}") from e
        finally:
            conn.close()

    def _load_folders(self, conn: sqlite3.Connection) -> dict[int, str]:
        """Load all note folders from the database.

        Args:
            conn: SQLite connection to NoteStore.sqlite.

        Returns:
            Mapping of folder Z_PK to folder title.
        """
        folders: dict[int, str] = {}
        try:
            cursor = conn.execute(
                "SELECT Z_PK, ZTITLE FROM ZICCLOUDSYNCINGOBJECT "
                "WHERE ZTITLE IS NOT NULL AND Z_ENT != 0"
            )
            for row in cursor:
                folders[row[0]] = row[1]
        except sqlite3.Error as e:
            logger.warning("Failed to load folders: %s", e)
        return folders

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
