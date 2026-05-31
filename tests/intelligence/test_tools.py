"""Unit tests for ToolRegistry — tool definitions and execution.

Tests verify:
- Tool definitions structure and completeness
- Execution routing and error handling
- Contact resolution and fallback logic
- Filter construction for each tool type
- Date parsing and result serialization
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from mudline.exceptions import SearchError
from mudline.intelligence.tools import ToolRegistry, _parse_iso_datetime, _result_to_dict
from mudline.models.document import Document, DocumentType, Source
from mudline.models.retriever import Filters, Result


def _make_source() -> Source:
    return Source(
        backup_id="test-backup",
        domain="HomeDomain",
        relative_path="test.db",
        backup_timestamp=datetime(2025, 1, 1),
    )


def _make_document(
    doc_type: DocumentType = DocumentType.MESSAGE,
    text: str = "Hello world",
    timestamp: datetime | None = None,
    metadata: dict | None = None,
) -> Document:
    return Document(
        type=doc_type,
        text=text,
        source=_make_source(),
        timestamp=timestamp or datetime(2025, 1, 15, 10, 30),
        metadata=metadata or {},
    )


def _make_result(
    doc_type: DocumentType = DocumentType.MESSAGE,
    text: str = "Hello world",
    score: float = 0.9,
    metadata: dict | None = None,
) -> Result:
    return Result(
        document=_make_document(doc_type=doc_type, text=text, metadata=metadata),
        score=score,
    )


def _make_registry(
    retriever: MagicMock | None = None,
    contact_index: MagicMock | None = None,
) -> tuple[ToolRegistry, MagicMock, MagicMock]:
    retriever = retriever or MagicMock()
    contact_index = contact_index or MagicMock()
    registry = ToolRegistry(retriever=retriever, contact_index=contact_index)
    return registry, retriever, contact_index


class TestToolDefs:
    """Test tool definition structure and completeness."""

    def test_get_tool_defs_returns_seven(self) -> None:
        """All 7 tool definitions are returned."""
        registry, _, _ = _make_registry()
        defs = registry.get_tool_defs()
        assert len(defs) == 7

    def test_tool_defs_have_valid_json_schema(self) -> None:
        """Each tool def has a valid JSON Schema with type: object and properties."""
        registry, _, _ = _make_registry()
        for td in registry.get_tool_defs():
            assert td.parameters["type"] == "object", f"{td.name} missing type: object"
            assert "properties" in td.parameters, f"{td.name} missing properties"

    def test_tool_defs_have_unique_names(self) -> None:
        """All tool names are unique."""
        registry, _, _ = _make_registry()
        names = [td.name for td in registry.get_tool_defs()]
        assert len(names) == len(set(names))

    def test_expected_tool_names(self) -> None:
        """All expected tool names are present."""
        registry, _, _ = _make_registry()
        names = {td.name for td in registry.get_tool_defs()}
        expected = {
            "search_messages",
            "search_notes",
            "search_photos",
            "get_contact",
            "get_call_history",
            "search_calendar",
            "search_safari",
        }
        assert names == expected

    def test_get_tool_defs_returns_copy(self) -> None:
        """Returned list is a copy — mutations don't affect internal state."""
        registry, _, _ = _make_registry()
        defs = registry.get_tool_defs()
        defs.clear()
        assert len(registry.get_tool_defs()) == 7


class TestExecuteRouting:
    """Test tool execution dispatch and error handling."""

    def test_unknown_tool_raises_search_error(self) -> None:
        """Executing an unknown tool raises SearchError."""
        registry, _, _ = _make_registry()
        with pytest.raises(SearchError, match="Unknown tool"):
            registry.execute("nonexistent_tool", {})

    def test_search_messages_calls_retriever(self) -> None:
        """search_messages routes to retriever.search with MESSAGE type."""
        registry, retriever, _ = _make_registry()
        retriever.search.return_value = [_make_result()]

        results = registry.execute("search_messages", {"query": "plumber"})

        retriever.search.assert_called_once()
        call_kwargs = retriever.search.call_args
        filters = (
            call_kwargs.kwargs.get("filters") or call_kwargs[1].get("filters") or call_kwargs[0][1]
        )
        assert DocumentType.MESSAGE in filters.data_types
        assert len(results) == 1

    def test_search_notes_calls_retriever(self) -> None:
        """search_notes routes to retriever.search with NOTE type."""
        registry, retriever, _ = _make_registry()
        retriever.search.return_value = []

        registry.execute("search_notes", {"query": "groceries"})

        retriever.search.assert_called_once()

    def test_search_photos_calls_retriever(self) -> None:
        """search_photos routes to retriever.search with PHOTO type."""
        registry, retriever, _ = _make_registry()
        retriever.search.return_value = []

        registry.execute("search_photos", {"query": "sunset"})

        retriever.search.assert_called_once()

    def test_search_calendar_calls_retriever(self) -> None:
        """search_calendar routes to retriever.search with CALENDAR type."""
        registry, retriever, _ = _make_registry()
        retriever.search.return_value = []

        registry.execute("search_calendar", {"query": "meeting"})

        retriever.search.assert_called_once()

    def test_search_safari_calls_retriever(self) -> None:
        """search_safari routes to retriever.search with SAFARI type."""
        registry, retriever, _ = _make_registry()
        retriever.search.return_value = []

        registry.execute("search_safari", {"query": "recipe"})

        retriever.search.assert_called_once()


class TestContactResolution:
    """Test contact name resolution via ContactIndex."""

    def test_search_messages_resolves_contact(self) -> None:
        """search_messages resolves contact name to handles via ContactIndex."""
        registry, retriever, contact_index = _make_registry()
        contact_index.resolve.return_value = ["+15551234567"]
        retriever.search.return_value = []

        registry.execute("search_messages", {"query": "hello", "contact": "Sarah"})

        contact_index.resolve.assert_called_once_with("Sarah")
        call_args = retriever.search.call_args
        filters = call_args.kwargs.get("filters") or call_args[1].get("filters") or call_args[0][1]
        assert filters.contacts == ["+15551234567"]

    def test_search_messages_fallback_raw_handle(self) -> None:
        """When resolve returns empty, raw value is used as handle."""
        registry, retriever, contact_index = _make_registry()
        contact_index.resolve.return_value = []
        retriever.search.return_value = []

        registry.execute("search_messages", {"query": "hello", "contact": "+15559999999"})

        call_args = retriever.search.call_args
        filters = call_args.kwargs.get("filters") or call_args[1].get("filters") or call_args[0][1]
        assert filters.contacts == ["+15559999999"]

    def test_search_messages_no_contact_no_resolution(self) -> None:
        """Without contact arg, no contact resolution is attempted."""
        registry, retriever, contact_index = _make_registry()
        retriever.search.return_value = []

        registry.execute("search_messages", {"query": "hello"})

        contact_index.resolve.assert_not_called()
        call_args = retriever.search.call_args
        filters = call_args.kwargs.get("filters") or call_args[1].get("filters") or call_args[0][1]
        assert filters.contacts is None

    def test_get_contact_resolves_and_searches(self) -> None:
        """get_contact resolves name then searches CONTACT type."""
        registry, retriever, contact_index = _make_registry()
        contact_index.resolve.return_value = ["+15551234567"]
        retriever.search.return_value = [
            _make_result(doc_type=DocumentType.CONTACT, text="Sarah Smith")
        ]

        results = registry.execute("get_contact", {"name": "Sarah"})

        contact_index.resolve.assert_called_once_with("Sarah")
        call_args = retriever.search.call_args
        filters = call_args.kwargs.get("filters") or call_args[1].get("filters") or call_args[0][1]
        assert DocumentType.CONTACT in filters.data_types
        assert filters.contacts == ["+15551234567"]
        assert len(results) == 1

    def test_get_contact_no_handles_returns_empty(self) -> None:
        """get_contact returns empty list when contact not found."""
        registry, retriever, contact_index = _make_registry()
        contact_index.resolve.return_value = []

        results = registry.execute("get_contact", {"name": "Unknown Person"})

        assert results == []
        retriever.search.assert_not_called()


class TestFilterConstruction:
    """Test that tools build correct Filters with metadata."""

    def test_search_notes_folder_filter(self) -> None:
        """search_notes passes folder as metadata filter."""
        registry, retriever, _ = _make_registry()
        retriever.search.return_value = []

        registry.execute("search_notes", {"query": "todo", "folder": "Work"})

        call_args = retriever.search.call_args
        filters = call_args.kwargs.get("filters") or call_args[1].get("filters") or call_args[0][1]
        assert filters.metadata == {"folder": "Work"}

    def test_get_call_history_call_type_filter(self) -> None:
        """get_call_history passes call_type as metadata filter."""
        registry, retriever, _ = _make_registry()
        retriever.search.return_value = []

        registry.execute("get_call_history", {"call_type": "missed"})

        call_args = retriever.search.call_args
        filters = call_args.kwargs.get("filters") or call_args[1].get("filters") or call_args[0][1]
        assert filters.metadata == {"call_type": "missed"}

    def test_call_history_resolves_contact(self) -> None:
        """get_call_history resolves contact via ContactIndex."""
        registry, retriever, contact_index = _make_registry()
        contact_index.resolve.return_value = ["+15551234567"]
        retriever.search.return_value = []

        registry.execute("get_call_history", {"contact": "John"})

        contact_index.resolve.assert_called_once_with("John")

    def test_date_filters_parsed(self) -> None:
        """after/before arguments are parsed into datetime Filters."""
        registry, retriever, _ = _make_registry()
        retriever.search.return_value = []

        registry.execute(
            "search_messages",
            {
                "query": "test",
                "after": "2025-01-01T00:00:00",
                "before": "2025-02-01T00:00:00",
            },
        )

        call_args = retriever.search.call_args
        filters = call_args.kwargs.get("filters") or call_args[1].get("filters") or call_args[0][1]
        assert filters.date_after == datetime(2025, 1, 1)
        assert filters.date_before == datetime(2025, 2, 1)

    def test_limit_passed_to_retriever(self) -> None:
        """Explicit limit argument is forwarded to retriever.search."""
        registry, retriever, _ = _make_registry()
        retriever.search.return_value = []

        registry.execute("search_messages", {"query": "test", "limit": 5})

        call_args = retriever.search.call_args
        limit = call_args.kwargs.get("limit") or call_args[1].get("limit") or call_args[0][2]
        assert limit == 5


class TestDateParsing:
    """Test ISO datetime parsing helper."""

    def test_valid_iso_date(self) -> None:
        """Valid ISO datetime string is parsed correctly."""
        result = _parse_iso_datetime("2025-01-15T10:30:00")
        assert result == datetime(2025, 1, 15, 10, 30, 0)

    def test_none_returns_none(self) -> None:
        """None input returns None."""
        assert _parse_iso_datetime(None) is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        assert _parse_iso_datetime("") is None

    def test_invalid_string_returns_none(self) -> None:
        """Invalid date string returns None with logged warning."""
        assert _parse_iso_datetime("not-a-date") is None


class TestResultToDict:
    """Test Result serialization."""

    def test_basic_serialization(self) -> None:
        """Result is correctly serialized to dict."""
        result = _make_result(text="test message", score=0.85, metadata={"handle": "+1555"})
        d = _result_to_dict(result)

        assert d["text"] == "test message"
        assert d["score"] == 0.85
        assert d["type"] == "message"
        assert d["timestamp"] == "2025-01-15T10:30:00"
        assert d["metadata"] == {"handle": "+1555"}
        assert "id" in d

    def test_none_timestamp(self) -> None:
        """Document without timestamp serializes timestamp as None."""
        doc = Document(
            type=DocumentType.CONTACT,
            text="John Doe",
            source=_make_source(),
            timestamp=None,
        )
        result = Result(document=doc, score=0.5)
        d = _result_to_dict(result)

        assert d["timestamp"] is None
