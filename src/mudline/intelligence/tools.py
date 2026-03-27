"""Tool definitions and execution for LLM function calling.

Defines the tool schemas that the LLM can invoke (search_messages, search_notes,
etc.) and maps each tool call to the appropriate Retriever query with filters.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from mudline.exceptions import SearchError
from mudline.index.contacts import ContactIndex
from mudline.intelligence.llm import ToolDef
from mudline.models.document import DocumentType
from mudline.models.retriever import Filters, Result, Retriever

logger = logging.getLogger(__name__)

_ISO_DATE_DESCRIPTION = "ISO 8601 datetime string (e.g. 2025-01-15T00:00:00)"

_SEARCH_MESSAGES_DEF = ToolDef(
    name="search_messages",
    description=(
        "Search text messages (iMessage/SMS). Use to find conversations, "
        "specific messages, or message history with a contact."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query for message content.",
            },
            "contact": {
                "type": "string",
                "description": "Contact name or phone/email handle to filter by.",
            },
            "after": {
                "type": "string",
                "description": f"Only messages after this date. {_ISO_DATE_DESCRIPTION}",
            },
            "before": {
                "type": "string",
                "description": f"Only messages before this date. {_ISO_DATE_DESCRIPTION}",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
                "default": 10,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)

_SEARCH_NOTES_DEF = ToolDef(
    name="search_notes",
    description=(
        "Search Apple Notes. Use to find notes by content, title, or folder."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query for note content.",
            },
            "folder": {
                "type": "string",
                "description": "Filter to a specific Notes folder.",
            },
            "after": {
                "type": "string",
                "description": f"Only notes after this date. {_ISO_DATE_DESCRIPTION}",
            },
            "before": {
                "type": "string",
                "description": f"Only notes before this date. {_ISO_DATE_DESCRIPTION}",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
                "default": 10,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)

_SEARCH_PHOTOS_DEF = ToolDef(
    name="search_photos",
    description=(
        "Search photos and videos by description, date, or location."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query (e.g. 'sunset at beach').",
            },
            "after": {
                "type": "string",
                "description": f"Only photos after this date. {_ISO_DATE_DESCRIPTION}",
            },
            "before": {
                "type": "string",
                "description": f"Only photos before this date. {_ISO_DATE_DESCRIPTION}",
            },
            "has_location": {
                "type": "boolean",
                "description": "If true, only return photos with GPS coordinates.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
                "default": 10,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)

_GET_CONTACT_DEF = ToolDef(
    name="get_contact",
    description=(
        "Look up a contact by name. Returns phone numbers, emails, and "
        "other info for the matching contact(s)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Contact name to search for (supports partial/fuzzy matching).",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
)

_GET_CALL_HISTORY_DEF = ToolDef(
    name="get_call_history",
    description="Retrieve phone call history, optionally filtered by contact or call type.",
    parameters={
        "type": "object",
        "properties": {
            "contact": {
                "type": "string",
                "description": "Contact name or phone handle to filter by.",
            },
            "call_type": {
                "type": "string",
                "enum": ["incoming", "outgoing", "missed"],
                "description": "Filter by call direction/status.",
            },
            "after": {
                "type": "string",
                "description": f"Only calls after this date. {_ISO_DATE_DESCRIPTION}",
            },
            "before": {
                "type": "string",
                "description": f"Only calls before this date. {_ISO_DATE_DESCRIPTION}",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
                "default": 10,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)

_SEARCH_CALENDAR_DEF = ToolDef(
    name="search_calendar",
    description="Search calendar events by content, date range, or attendees.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query for event content.",
            },
            "after": {
                "type": "string",
                "description": f"Only events after this date. {_ISO_DATE_DESCRIPTION}",
            },
            "before": {
                "type": "string",
                "description": f"Only events before this date. {_ISO_DATE_DESCRIPTION}",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
                "default": 10,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)

_SEARCH_SAFARI_DEF = ToolDef(
    name="search_safari",
    description="Search Safari browsing history and bookmarks.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query for page titles or URLs.",
            },
            "after": {
                "type": "string",
                "description": f"Only history after this date. {_ISO_DATE_DESCRIPTION}",
            },
            "before": {
                "type": "string",
                "description": f"Only history before this date. {_ISO_DATE_DESCRIPTION}",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
                "default": 10,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)

_ALL_TOOL_DEFS = [
    _SEARCH_MESSAGES_DEF,
    _SEARCH_NOTES_DEF,
    _SEARCH_PHOTOS_DEF,
    _GET_CONTACT_DEF,
    _GET_CALL_HISTORY_DEF,
    _SEARCH_CALENDAR_DEF,
    _SEARCH_SAFARI_DEF,
]


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string, returning None on invalid input."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        logger.warning("Failed to parse datetime: %s", value)
        return None


def _result_to_dict(result: Result) -> dict[str, Any]:
    """Convert a Result to a JSON-serializable dict."""
    doc = result.document
    return {
        "id": doc.id,
        "text": doc.text,
        "timestamp": doc.timestamp.isoformat() if doc.timestamp else None,
        "type": doc.type.value,
        "metadata": doc.metadata,
        "score": result.score,
    }


class ToolRegistry:
    """Registry of tools available for LLM function calling.

    Maps tool names to their JSON Schema definitions and execution logic.
    Each tool translates LLM-provided arguments into Retriever calls with
    appropriate Filters.

    Args:
        retriever: The search/retrieval backend.
        contact_index: Contact resolution index for name-to-handle mapping.
    """

    def __init__(self, retriever: Retriever, contact_index: ContactIndex) -> None:
        self._retriever = retriever
        self._contact_index = contact_index
        self._executors: dict[str, Any] = {
            "search_messages": self._search_messages,
            "search_notes": self._search_notes,
            "search_photos": self._search_photos,
            "get_contact": self._get_contact,
            "get_call_history": self._get_call_history,
            "search_calendar": self._search_calendar,
            "search_safari": self._search_safari,
        }

    def get_tool_defs(self) -> list[ToolDef]:
        """Return all tool definitions for LLM function calling."""
        return list(_ALL_TOOL_DEFS)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute a tool call and return structured results.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments dict from the LLM's tool call.

        Returns:
            List of dicts with keys: id, text, timestamp, type, metadata, score.

        Raises:
            SearchError: If the tool name is unknown or the search backend fails.
        """
        executor = self._executors.get(tool_name)
        if executor is None:
            raise SearchError(f"Unknown tool: {tool_name}")

        logger.info("Executing tool %s with args: %s", tool_name, arguments)
        return executor(arguments)

    def _resolve_contact_handles(self, contact: str | None) -> list[str] | None:
        """Resolve a contact name/handle to a list of handles via ContactIndex."""
        if not contact:
            return None
        handles = self._contact_index.resolve(contact)
        if handles:
            return handles
        # If resolve found nothing, treat the raw value as a handle itself
        return [contact]

    def _build_filters(
        self,
        args: dict[str, Any],
        data_types: list[DocumentType],
        *,
        resolve_contact: bool = False,
    ) -> Filters:
        """Build a Filters object from common tool arguments."""
        contacts = None
        if resolve_contact:
            contacts = self._resolve_contact_handles(args.get("contact"))

        metadata = None
        if "folder" in args and args["folder"]:
            metadata = {"folder": args["folder"]}
        if "call_type" in args and args["call_type"]:
            metadata = metadata or {}
            metadata["call_type"] = args["call_type"]
        if args.get("has_location"):
            metadata = metadata or {}
            metadata["has_location"] = "true"

        return Filters(
            data_types=data_types,
            contacts=contacts,
            date_after=_parse_iso_datetime(args.get("after")),
            date_before=_parse_iso_datetime(args.get("before")),
            metadata=metadata,
        )

    def _do_search(
        self,
        args: dict[str, Any],
        data_types: list[DocumentType],
        *,
        resolve_contact: bool = False,
    ) -> list[dict[str, Any]]:
        """Common search pattern: build filters, call retriever, format results."""
        filters = self._build_filters(args, data_types, resolve_contact=resolve_contact)
        limit = args.get("limit", 10)
        query = args.get("query")
        results = self._retriever.search(query=query, filters=filters, limit=limit)
        return [_result_to_dict(r) for r in results]

    def _search_messages(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        return self._do_search(args, [DocumentType.MESSAGE], resolve_contact=True)

    def _search_notes(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        return self._do_search(args, [DocumentType.NOTE])

    def _search_photos(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        return self._do_search(args, [DocumentType.PHOTO])

    def _get_contact(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        """Resolve a contact name and return matching contact documents."""
        name = args.get("name", "")
        handles = self._contact_index.resolve(name)
        if not handles:
            return []

        filters = Filters(
            data_types=[DocumentType.CONTACT],
            contacts=handles,
        )
        results = self._retriever.search(query=None, filters=filters, limit=10)
        return [_result_to_dict(r) for r in results]

    def _get_call_history(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        return self._do_search(args, [DocumentType.CALL], resolve_contact=True)

    def _search_calendar(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        return self._do_search(args, [DocumentType.CALENDAR])

    def _search_safari(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        return self._do_search(args, [DocumentType.SAFARI])
