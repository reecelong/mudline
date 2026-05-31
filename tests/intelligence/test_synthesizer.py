"""Unit tests for Synthesizer — natural language answer generation with citations.

Tests verify:
- Empty evidence returns "couldn't find" message
- Evidence formatting and LLM invocation
- Citation building from dict and Document sources
- Module-level formatting functions for each document type
- Contact extraction logic
- Expanded result formatting with context markers
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mudline.intelligence.context import ExpandedResult
from mudline.intelligence.llm import LLMResponse
from mudline.intelligence.synthesizer import (
    Citation,
    Synthesizer,
    SynthesizedAnswer,
    _extract_contact,
    _format_document,
    _format_expanded_result,
    _format_result_dict,
)
from mudline.models.document import Document, DocumentType, Source
from mudline.models.retriever import Result


def _make_source() -> Source:
    return Source(
        backup_id="test-backup",
        domain="HomeDomain",
        relative_path="test.db",
        backup_timestamp=datetime(2025, 1, 1),
    )


def _make_document(
    doc_type: DocumentType = DocumentType.MESSAGE,
    text: str = "Hello",
    timestamp: datetime | None = None,
    metadata: dict | None = None,
    doc_id: str = "",
) -> Document:
    doc = Document(
        type=doc_type,
        text=text,
        source=_make_source(),
        timestamp=timestamp or datetime(2025, 1, 15, 10, 30),
        metadata=metadata or {},
    )
    if doc_id:
        object.__setattr__(doc, "id", doc_id)
    return doc


def _make_plan_result(tool_results: dict | None = None):
    """Create a mock PlanResult-like object."""
    return SimpleNamespace(tool_results=tool_results or {})


class TestSynthesizeEmptyEvidence:
    """Test synthesize with no evidence."""

    @pytest.mark.asyncio
    async def test_empty_tool_results_returns_not_found(self) -> None:
        """Empty tool results returns the 'couldn't find' message."""
        llm = AsyncMock()
        synth = Synthesizer(llm=llm)

        plan = _make_plan_result(tool_results={})
        answer = await synth.synthesize("what did Sarah say?", plan)

        assert isinstance(answer, SynthesizedAnswer)
        assert "couldn't find" in answer.text.lower()
        assert answer.citations == []
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_results_lists_returns_not_found(self) -> None:
        """Tool results with empty lists returns the 'couldn't find' message."""
        llm = AsyncMock()
        synth = Synthesizer(llm=llm)

        plan = _make_plan_result(tool_results={"tc1": []})
        answer = await synth.synthesize("what did Sarah say?", plan)

        assert "couldn't find" in answer.text.lower()


class TestSynthesizeWithEvidence:
    """Test synthesize with tool results and expanded results."""

    @pytest.mark.asyncio
    async def test_formats_evidence_and_calls_llm(self) -> None:
        """Tool results are formatted as evidence and sent to LLM."""
        llm = AsyncMock()
        llm.complete.return_value = LLMResponse(
            content="Sarah mentioned the plumber on Jan 15.",
            model="test",
            usage={"input_tokens": 50, "output_tokens": 20},
        )

        synth = Synthesizer(llm=llm)
        plan = _make_plan_result(
            tool_results={
                "tc1": [
                    {
                        "id": "msg1",
                        "text": "the plumber is coming tomorrow",
                        "timestamp": "2025-01-15T10:30:00",
                        "type": "message",
                        "metadata": {"handle": "+15551234567", "is_from_me": False},
                        "score": 0.9,
                    },
                ],
            }
        )

        answer = await synth.synthesize("what did Sarah say about the plumber?", plan)

        assert answer.text == "Sarah mentioned the plumber on Jan 15."
        assert answer.usage["input_tokens"] == 50
        llm.complete.assert_called_once()

        # Verify evidence was included in the prompt
        call_args = llm.complete.call_args
        messages = call_args.kwargs.get("messages", call_args[0][0] if call_args[0] else [])
        assert any("plumber" in m.content for m in messages)

    @pytest.mark.asyncio
    async def test_citations_built_from_dict_results(self) -> None:
        """Citations are created from serialized dict tool results."""
        llm = AsyncMock()
        llm.complete.return_value = LLMResponse(content="answer", model="test")

        synth = Synthesizer(llm=llm)
        plan = _make_plan_result(
            tool_results={
                "tc1": [
                    {
                        "id": "doc1",
                        "text": "message text here",
                        "timestamp": "2025-01-15T10:30:00",
                        "type": "message",
                        "metadata": {"handle": "+1555"},
                        "score": 0.9,
                    },
                ],
            }
        )

        answer = await synth.synthesize("query", plan)

        assert len(answer.citations) == 1
        c = answer.citations[0]
        assert c.document_id == "doc1"
        assert c.document_type == "message"
        assert c.timestamp == "2025-01-15T10:30:00"
        assert c.contact == "+1555"
        assert "message text" in c.excerpt

    @pytest.mark.asyncio
    async def test_citations_built_from_expanded_results(self) -> None:
        """Citations are created from Document objects in expanded results."""
        llm = AsyncMock()
        llm.complete.return_value = LLMResponse(content="answer", model="test")

        doc = _make_document(
            doc_type=DocumentType.MESSAGE,
            text="expanded message",
            timestamp=datetime(2025, 1, 15, 10, 30),
            metadata={"handle": "+1555"},
            doc_id="edoc1",
        )
        ctx_doc = _make_document(
            text="context msg",
            timestamp=datetime(2025, 1, 15, 10, 25),
            metadata={"handle": "+1555"},
            doc_id="ctx1",
        )

        er = ExpandedResult(
            result=Result(document=doc, score=0.9),
            context_before=[ctx_doc],
        )

        synth = Synthesizer(llm=llm)
        plan = _make_plan_result(tool_results={})
        answer = await synth.synthesize("query", plan, expanded_results=[er])

        # Should have citations from both the primary and context docs
        ids = {c.document_id for c in answer.citations}
        assert "edoc1" in ids
        assert "ctx1" in ids


class TestFormatResultDict:
    """Test _format_result_dict for each document type."""

    def test_message_format(self) -> None:
        """Message dict formats as [timestamp] Direction handle: text."""
        item = {
            "timestamp": "2025-01-15T10:30:00",
            "text": "hello there",
            "type": "message",
            "metadata": {"handle": "+1555", "is_from_me": False},
        }
        result = _format_result_dict(item)
        assert "From" in result
        assert "+1555" in result
        assert "hello there" in result

    def test_message_from_me(self) -> None:
        """Message from_me uses 'To' direction."""
        item = {
            "timestamp": "2025-01-15",
            "text": "hi",
            "type": "message",
            "metadata": {"handle": "+1555", "is_from_me": True},
        }
        result = _format_result_dict(item)
        assert "To" in result

    def test_call_format(self) -> None:
        """Call dict formats with type, handle, and duration."""
        item = {
            "timestamp": "2025-01-15",
            "text": "",
            "type": "call",
            "metadata": {"call_type": "incoming", "handle": "+1555", "duration_seconds": 125},
        }
        result = _format_result_dict(item)
        assert "Incoming" in result
        assert "+1555" in result
        assert "2m 5s" in result

    def test_note_format(self) -> None:
        """Note dict formats with optional folder."""
        item = {"type": "note", "text": "buy milk", "metadata": {"folder": "Shopping"}}
        result = _format_result_dict(item)
        assert "Note (Shopping)" in result
        assert "buy milk" in result

    def test_note_no_folder(self) -> None:
        """Note without folder omits parenthetical."""
        item = {"type": "note", "text": "reminder", "metadata": {}}
        result = _format_result_dict(item)
        assert result.startswith("Note:")

    def test_contact_format(self) -> None:
        """Contact dict formats as 'Contact: text'."""
        item = {"type": "contact", "text": "John Doe", "metadata": {}}
        assert _format_result_dict(item) == "Contact: John Doe"

    def test_photo_format(self) -> None:
        """Photo dict formats with timestamp and optional album."""
        item = {
            "timestamp": "2025-01-15",
            "text": "",
            "type": "photo",
            "metadata": {"album": "Vacation"},
        }
        result = _format_result_dict(item)
        assert "Photo" in result
        assert "Vacation" in result

    def test_calendar_format(self) -> None:
        """Calendar dict formats event with optional location."""
        item = {
            "timestamp": "2025-01-15",
            "text": "Team standup",
            "type": "calendar",
            "metadata": {"location": "Room 5"},
        }
        result = _format_result_dict(item)
        assert "Event: Team standup" in result
        assert "Room 5" in result

    def test_safari_format(self) -> None:
        """Safari dict formats with URL."""
        item = {
            "timestamp": "2025-01-15",
            "text": "Example Page",
            "type": "safari",
            "metadata": {"url": "https://example.com"},
        }
        result = _format_result_dict(item)
        assert "Safari" in result
        assert "https://example.com" in result

    def test_unknown_type_format(self) -> None:
        """Unknown type falls through to generic format."""
        item = {"timestamp": "2025-01-15", "text": "something", "type": "widget", "metadata": {}}
        result = _format_result_dict(item)
        assert "widget" in result
        assert "something" in result


class TestFormatDocument:
    """Test _format_document for Document objects."""

    def test_message_document(self) -> None:
        """Message Document formats with direction, handle, text."""
        doc = _make_document(
            doc_type=DocumentType.MESSAGE,
            text="hey there",
            metadata={"handle": "+1555", "is_from_me": False},
        )
        result = _format_document(doc)
        assert "From" in result
        assert "+1555" in result
        assert "hey there" in result

    def test_note_document(self) -> None:
        """Note Document formats with folder."""
        doc = _make_document(
            doc_type=DocumentType.NOTE,
            text="buy eggs",
            metadata={"folder": "Lists"},
        )
        result = _format_document(doc)
        assert "Note (Lists)" in result

    def test_call_document(self) -> None:
        """Call Document formats with type, handle, duration."""
        doc = _make_document(
            doc_type=DocumentType.CALL,
            text="",
            metadata={"call_type": "outgoing", "handle": "+1555", "duration_seconds": 62},
        )
        result = _format_document(doc)
        assert "Outgoing" in result
        assert "1m 2s" in result

    def test_voicemail_document(self) -> None:
        """Voicemail Document formats with handle and transcription."""
        doc = _make_document(
            doc_type=DocumentType.VOICEMAIL,
            text="original text",
            metadata={"handle": "+1555", "duration_seconds": 30, "transcription": "hi there"},
        )
        result = _format_document(doc)
        assert "Voicemail" in result
        assert "hi there" in result

    def test_photo_document(self) -> None:
        """Photo Document formats with location and album."""
        doc = _make_document(
            doc_type=DocumentType.PHOTO,
            text="sunset",
            metadata={"latitude": 37.7749, "longitude": -122.4194, "album": "Travel"},
        )
        result = _format_document(doc)
        assert "Photo" in result
        assert "37.7749" in result
        assert "Travel" in result
        assert "sunset" in result

    def test_calendar_document(self) -> None:
        """Calendar Document formats event with location."""
        doc = _make_document(
            doc_type=DocumentType.CALENDAR,
            text="Dentist",
            metadata={"location": "123 Main St"},
        )
        result = _format_document(doc)
        assert "Event: Dentist" in result
        assert "123 Main St" in result

    def test_contact_document(self) -> None:
        """Contact Document formats as 'Contact: text'."""
        doc = _make_document(doc_type=DocumentType.CONTACT, text="Jane Smith")
        assert _format_document(doc) == "Contact: Jane Smith"

    def test_safari_document(self) -> None:
        """Safari Document formats with URL."""
        doc = _make_document(
            doc_type=DocumentType.SAFARI,
            text="Example",
            metadata={"url": "https://example.com"},
        )
        result = _format_document(doc)
        assert "Safari" in result
        assert "https://example.com" in result

    def test_no_timestamp(self) -> None:
        """Document without timestamp shows 'unknown date'."""
        doc = Document(
            type=DocumentType.CONTACT,
            text="Someone",
            source=_make_source(),
            timestamp=None,
        )
        result = _format_document(doc)
        assert "Contact: Someone" in result


class TestExtractContact:
    """Test _extract_contact for various document types."""

    def test_message_returns_handle(self) -> None:
        """MESSAGE type extracts handle from metadata."""
        doc = _make_document(doc_type=DocumentType.MESSAGE, metadata={"handle": "+1555"})
        assert _extract_contact(doc) == "+1555"

    def test_call_returns_handle(self) -> None:
        """CALL type extracts handle from metadata."""
        doc = _make_document(doc_type=DocumentType.CALL, metadata={"handle": "+1555"})
        assert _extract_contact(doc) == "+1555"

    def test_voicemail_returns_handle(self) -> None:
        """VOICEMAIL type extracts handle from metadata."""
        doc = _make_document(doc_type=DocumentType.VOICEMAIL, metadata={"handle": "+1555"})
        assert _extract_contact(doc) == "+1555"

    def test_calendar_returns_attendees(self) -> None:
        """CALENDAR type extracts attendees (up to 3, joined)."""
        doc = _make_document(
            doc_type=DocumentType.CALENDAR,
            metadata={"attendees": ["Alice", "Bob", "Charlie", "Dave"]},
        )
        result = _extract_contact(doc)
        assert result == "Alice, Bob, Charlie"

    def test_calendar_no_attendees_returns_none(self) -> None:
        """CALENDAR without attendees returns None."""
        doc = _make_document(doc_type=DocumentType.CALENDAR, metadata={})
        assert _extract_contact(doc) is None

    def test_note_returns_none(self) -> None:
        """NOTE type returns None."""
        doc = _make_document(doc_type=DocumentType.NOTE)
        assert _extract_contact(doc) is None

    def test_photo_returns_none(self) -> None:
        """PHOTO type returns None."""
        doc = _make_document(doc_type=DocumentType.PHOTO)
        assert _extract_contact(doc) is None


class TestFormatExpandedResult:
    """Test _format_expanded_result with context markers."""

    def test_with_context_adds_markers(self) -> None:
        """Expanded result with context adds conversation markers."""
        pivot = _make_document(text="pivot msg", doc_id="p")
        before = _make_document(
            text="before msg",
            timestamp=datetime(2025, 1, 15, 10, 0),
            doc_id="b",
        )
        after = _make_document(
            text="after msg",
            timestamp=datetime(2025, 1, 15, 11, 0),
            doc_id="a",
        )

        er = ExpandedResult(
            result=Result(document=pivot, score=0.9),
            context_before=[before],
            context_after=[after],
        )
        result = _format_expanded_result(er)

        assert "--- Conversation context ---" in result
        assert ">>>" in result
        assert "<<<" in result
        assert "before msg" in result
        assert "after msg" in result

    def test_without_context_no_markers(self) -> None:
        """Expanded result without context omits conversation markers."""
        pivot = _make_document(text="solo msg", doc_id="p")
        er = ExpandedResult(
            result=Result(document=pivot, score=0.9),
        )
        result = _format_expanded_result(er)

        assert "--- Conversation context ---" not in result
        assert ">>>" in result
