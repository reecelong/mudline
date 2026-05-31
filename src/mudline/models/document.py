"""Core document model — the universal interchange format between extraction and indexing.

Every extractor produces Document objects. The index layer consumes them.
DO NOT modify this file without updating all extractors and the ingest pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path


class DocumentType(StrEnum):
    """Supported document types.

    Most members map to iOS data sources. TRANSCRIPT is a generic,
    source-agnostic type for downstream engine consumers (e.g. audio/call
    transcripts) so they can produce Documents without forking this enum.
    """
    MESSAGE = "message"
    PHOTO = "photo"
    NOTE = "note"
    CONTACT = "contact"
    CALENDAR = "calendar"
    CALL = "call"
    VOICEMAIL = "voicemail"
    SAFARI = "safari"
    TRANSCRIPT = "transcript"


@dataclass(frozen=True)
class Source:
    """Provenance tracking — where this document came from."""
    backup_id: str            # UDID + backup timestamp
    domain: str               # iOS backup domain (e.g., "HomeDomain")
    relative_path: str        # Path within the domain
    backup_timestamp: datetime # When the backup was taken


@dataclass
class Attachment:
    """Reference to a binary file (photo, audio, etc.)."""
    filename: str
    mime_type: str
    path: Path | None = None   # Resolved path in backup, None if unresolvable
    size_bytes: int | None = None


@dataclass
class Document:
    """Universal document model produced by all extractors.

    Args:
        type: The kind of iOS data this represents.
        text: Primary text content for embedding and search.
              For messages: the message body.
              For photos: EXIF description or empty string.
              For notes: full note text.
              For contacts: "FirstName LastName — org — phone — email".
              For calendar: "Title — location — notes".
              For calls: "Call with +15551234567, 3m 42s, outgoing".
        timestamp: When this data was created/occurred. None for contacts.
        metadata: Type-specific structured data. Keys vary by DocumentType.
        source: Backup provenance.
        attachments: Binary file references.
        id: Deterministic unique identifier. Auto-generated if not provided.
    """
    type: DocumentType
    text: str
    source: Source
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[Attachment] = field(default_factory=list)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            # Deterministic ID from source + type + distinguishing content
            key = (
                f"{self.source.backup_id}:{self.source.domain}"
                f":{self.source.relative_path}:{self.type.value}"
            )
            if self.timestamp:
                key += f":{self.timestamp.isoformat()}"
            if self.text:
                key += f":{self.text[:100]}"
            object.__setattr__(self, "id", hashlib.sha256(key.encode()).hexdigest()[:16])


# --- Metadata key conventions per DocumentType ---
#
# MESSAGE:
#   handle: str           — phone number or email of the other party
#   is_from_me: bool
#   chat_id: int          — conversation thread ID
#   chat_name: str | None — group chat display name
#   participants: list[str] — all handles in the chat
#
# PHOTO:
#   latitude: float | None
#   longitude: float | None
#   width: int
#   height: int
#   album: str | None
#   media_type: str       — "image" or "video"
#
# NOTE:
#   folder: str | None    — folder/subfolder path
#   has_attachments: bool
#
# CONTACT:
#   phones: list[str]
#   emails: list[str]
#   organization: str | None
#   handles: list[str]    — all known handles for this contact
#
# CALENDAR:
#   location: str | None
#   attendees: list[str]
#   is_all_day: bool
#   end_time: datetime | None
#   recurrence: str | None
#
# CALL:
#   handle: str
#   duration_seconds: int
#   call_type: str        — "incoming" | "outgoing" | "missed"
#
# VOICEMAIL:
#   handle: str
#   duration_seconds: int
#   transcription: str | None
#
# SAFARI:
#   url: str
#   visit_count: int | None  — for history
#   folder: str | None        — for bookmarks
