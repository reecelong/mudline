"""MessageExtractor — extract iMessage and SMS from iOS backups.

Parses HomeDomain/Library/SMS/sms.db to extract messages with:
- Text content
- Timestamp (Cocoa epoch nanoseconds → datetime)
- is_from_me flag
- Handle (phone/email of the sender or recipient)
- Chat display name (for group conversations)
- Thread grouping by chat_id
- Attachment references

iOS message timestamps are stored as nanoseconds since Cocoa epoch (2001-01-01).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from mudline.exceptions import ExtractionError
from mudline.models.document import Attachment, Document, DocumentType, Source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mudline.models import ResourceResolver  # noqa: F401

logger = logging.getLogger(__name__)

COCOA_EPOCH = datetime(2001, 1, 1)


def _cocoa_ns_to_datetime(cocoa_ns: int) -> datetime:
    """Convert Cocoa epoch nanoseconds to datetime.

    Args:
        cocoa_ns: Nanoseconds since 2001-01-01.

    Returns:
        Converted datetime.
    """
    cocoa_seconds = cocoa_ns / 1e9
    return COCOA_EPOCH + timedelta(seconds=cocoa_seconds)


class MessageExtractor:
    """Extract messages from iOS SMS/iMessage database."""

    @property
    def domain(self) -> str:
        """iOS backup domain for messages."""
        return "HomeDomain"

    @property
    def data_type(self) -> str:
        """Document type produced by this extractor."""
        return DocumentType.MESSAGE.value

    def can_extract(self, resolver: ResourceResolver) -> bool:
        """Check if the SMS database exists in the backup.

        Args:
            resolver: ResourceResolver for the target backup.

        Returns:
            True if sms.db exists, False otherwise.
        """
        return resolver.file_exists(self.domain, "Library/SMS/sms.db")

    def extract(self, resolver: ResourceResolver) -> Iterator[Document]:
        """Extract all messages from the SMS database.

        Args:
            resolver: ResourceResolver for the target backup.

        Yields:
            Document objects for each message.

        Raises:
            FileNotFoundError: If sms.db is missing.
            ExtractionError: If the database schema is unexpected.
        """
        try:
            sms_path = resolver.resolve(self.domain, "Library/SMS/sms.db")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"SMS database not found in backup: {self.domain}/Library/SMS/sms.db"
            ) from e

        # Connect to the SMS database in read-only mode
        try:
            conn = sqlite3.connect(f"file:{sms_path}?mode=ro&immutable=1", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise ExtractionError(f"Failed to open SMS database: {sms_path}") from e

        try:
            # Build a map of handle IDs to handle values
            handles = self._load_handles(conn)

            # Build a map of chat IDs to chat names
            chat_names = self._load_chat_names(conn)

            # Build a map of message IDs to list of attachment IDs
            message_attachments = self._load_message_attachments(conn)

            # Fetch all messages
            cursor = conn.execute(
                """
                SELECT
                    m.ROWID as message_id,
                    m.text,
                    m.handle_id,
                    m.date,
                    m.is_from_me,
                    cmj.chat_id
                FROM message m
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                ORDER BY m.date ASC
                """
            )

            backup_id = self._build_backup_id(resolver)
            backup_timestamp = self._get_backup_timestamp(resolver)

            for row in cursor:
                message_id = row["message_id"]
                text = row["text"] or ""
                handle_id = row["handle_id"]
                date_ns = row["date"]
                is_from_me = bool(row["is_from_me"])
                chat_id = row["chat_id"]

                # Convert timestamp
                try:
                    timestamp = _cocoa_ns_to_datetime(date_ns)
                except (ValueError, OverflowError):
                    logger.warning(
                        "Invalid timestamp for message %d: %s, skipping",
                        message_id,
                        date_ns,
                    )
                    continue

                # Get handle
                handle = handles.get(handle_id, f"<unknown:{handle_id}>")

                # Get chat display name and participants
                chat_name = None
                participants = [handle]
                if chat_id is not None:
                    chat_name = chat_names.get(chat_id)
                    # For group chats, try to get all participants
                    participants = self._get_chat_participants(conn, chat_id, handles)

                # Get attachments for this message
                attachments = []
                if message_id in message_attachments:
                    attachment_ids = message_attachments[message_id]
                    attachments = self._load_attachments_for_message(conn, attachment_ids)

                # Build metadata
                metadata = {
                    "handle": handle,
                    "is_from_me": is_from_me,
                    "chat_id": chat_id,
                    "chat_name": chat_name,
                    "participants": participants,
                }

                # Create source
                source = Source(
                    backup_id=backup_id,
                    domain=self.domain,
                    relative_path=f"Library/SMS/sms.db/message/{message_id}",
                    backup_timestamp=backup_timestamp,
                )

                # Create document
                doc = Document(
                    type=DocumentType.MESSAGE,
                    text=text,
                    timestamp=timestamp,
                    metadata=metadata,
                    source=source,
                    attachments=attachments,
                )

                yield doc

        except sqlite3.Error as e:
            raise ExtractionError(f"Database error while extracting messages: {e}") from e
        finally:
            conn.close()

    def _load_handles(self, conn: sqlite3.Connection) -> dict[int, str]:
        """Load all handles (phone numbers/emails) from the database.

        Args:
            conn: SQLite connection to sms.db.

        Returns:
            Mapping of handle ROWID to handle ID string (phone/email).
        """
        handles: dict[int, str] = {}
        try:
            cursor = conn.execute("SELECT ROWID, id FROM handle")
            for row in cursor:
                handles[row[0]] = row[1]
        except sqlite3.Error as e:
            logger.warning("Failed to load handles: %s", e)
        return handles

    def _load_chat_names(self, conn: sqlite3.Connection) -> dict[int, str]:
        """Load all chat display names.

        Args:
            conn: SQLite connection to sms.db.

        Returns:
            Mapping of chat ROWID to display_name.
        """
        chat_names: dict[int, str] = {}
        try:
            cursor = conn.execute("SELECT ROWID, display_name FROM chat")
            for row in cursor:
                chat_id, display_name = row[0], row[1]
                if display_name:
                    chat_names[chat_id] = display_name
        except sqlite3.Error as e:
            logger.warning("Failed to load chat names: %s", e)
        return chat_names

    def _load_message_attachments(self, conn: sqlite3.Connection) -> dict[int, list[int]]:
        """Load mapping of message IDs to attachment IDs.

        Args:
            conn: SQLite connection to sms.db.

        Returns:
            Mapping of message ROWID to list of attachment ROWIDs.
        """
        message_attachments: dict[int, list[int]] = {}
        try:
            cursor = conn.execute("SELECT message_id, attachment_id FROM message_attachment_join")
            for row in cursor:
                message_id, attachment_id = row[0], row[1]
                if message_id not in message_attachments:
                    message_attachments[message_id] = []
                message_attachments[message_id].append(attachment_id)
        except sqlite3.Error as e:
            logger.warning("Failed to load message attachments: %s", e)
        return message_attachments

    def _load_attachments_for_message(
        self, conn: sqlite3.Connection, attachment_ids: list[int]
    ) -> list[Attachment]:
        """Load attachment metadata for a message.

        Args:
            conn: SQLite connection to sms.db.
            attachment_ids: List of attachment ROWIDs.

        Returns:
            List of Attachment objects.
        """
        attachments: list[Attachment] = []
        try:
            placeholders = ",".join("?" * len(attachment_ids))
            cursor = conn.execute(
                f"SELECT ROWID, filename, mime_type, total_bytes FROM attachment "
                f"WHERE ROWID IN ({placeholders})",
                attachment_ids,
            )
            for row in cursor:
                _, filename, mime_type, size_bytes = row
                attachments.append(
                    Attachment(
                        filename=filename or "unknown",
                        mime_type=mime_type or "application/octet-stream",
                        size_bytes=size_bytes,
                    )
                )
        except sqlite3.Error as e:
            logger.warning("Failed to load attachments: %s", e)
        return attachments

    def _get_chat_participants(
        self, conn: sqlite3.Connection, chat_id: int, handles: dict[int, str]
    ) -> list[str]:
        """Get all participant handles for a chat.

        Args:
            conn: SQLite connection to sms.db.
            chat_id: The chat ROWID.
            handles: Mapping of handle IDs to handle strings.

        Returns:
            List of unique handle strings in the chat.
        """
        participants: set[str] = set()
        try:
            cursor = conn.execute(
                """
                SELECT DISTINCT m.handle_id
                FROM message m
                JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                WHERE cmj.chat_id = ?
                """,
                (chat_id,),
            )
            for row in cursor:
                handle_id = row[0]
                if handle_id is not None and handle_id in handles:
                    participants.add(handles[handle_id])
        except sqlite3.Error as e:
            logger.warning("Failed to get chat participants for chat %d: %s", chat_id, e)
        return sorted(participants)

    def _build_backup_id(self, resolver: ResourceResolver) -> str:
        """Build a backup ID from backup path.

        Args:
            resolver: ResourceResolver.

        Returns:
            A string identifier for the backup.
        """
        # Use the backup directory name as the backup_id
        backup_name = resolver.backup_path.name
        return str(backup_name)

    def _get_backup_timestamp(self, resolver: ResourceResolver) -> datetime:
        """Get the backup timestamp from Info.plist or Status.plist.

        Args:
            resolver: ResourceResolver.

        Returns:
            Backup timestamp, or current time if unavailable.
        """
        import plistlib

        # Try Info.plist first
        info_plist = resolver.backup_path / "Info.plist"
        if info_plist.exists():
            try:
                with open(info_plist, "rb") as f:
                    info: Any = plistlib.load(f)
                    if "Last Backup Date" in info:
                        timestamp = info["Last Backup Date"]
                        if isinstance(timestamp, datetime):
                            return timestamp
            except Exception as e:
                logger.warning("Failed to read Info.plist: %s", e)

        # Fallback to Status.plist
        status_plist = resolver.backup_path / "Status.plist"
        if status_plist.exists():
            try:
                with open(status_plist, "rb") as f:
                    status: Any = plistlib.load(f)
                    if "Date" in status:
                        timestamp = status["Date"]
                        if isinstance(timestamp, datetime):
                            return timestamp
            except Exception as e:
                logger.warning("Failed to read Status.plist: %s", e)

        # Fallback to current time
        return datetime.now()
