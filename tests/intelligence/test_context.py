"""Unit tests for ContextExpander — search result context enrichment.

Tests verify:
- No-expansion types return empty context
- Message expansion via get_thread with before/after splitting
- Temporal expansion for photos, calendar, calls
- Missing metadata/timestamp edge cases
- Batch deduplication of shared context
- _split_around chronological splitting
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from mudline.intelligence.context import ContextExpander, ExpandedResult
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


def _make_result(
    doc_type: DocumentType = DocumentType.MESSAGE,
    text: str = "Hello",
    timestamp: datetime | None = None,
    metadata: dict | None = None,
    doc_id: str = "",
) -> Result:
    return Result(
        document=_make_document(doc_type, text, timestamp, metadata, doc_id),
        score=0.9,
    )


class TestNoExpansionTypes:
    """Types that are self-contained return empty context."""

    def test_note_returns_empty_context(self) -> None:
        """NOTE type returns ExpandedResult with no context."""
        retriever = MagicMock()
        expander = ContextExpander(retriever)
        result = _make_result(doc_type=DocumentType.NOTE, text="My note")

        expanded = expander.expand(result)

        assert expanded.result is result
        assert expanded.context_before == []
        assert expanded.context_after == []
        retriever.search.assert_not_called()

    def test_contact_returns_empty_context(self) -> None:
        """CONTACT type returns ExpandedResult with no context."""
        retriever = MagicMock()
        expander = ContextExpander(retriever)
        result = _make_result(doc_type=DocumentType.CONTACT, text="John Doe")

        expanded = expander.expand(result)

        assert expanded.context_before == []
        assert expanded.context_after == []

    def test_safari_returns_empty_context(self) -> None:
        """SAFARI type returns ExpandedResult with no context."""
        retriever = MagicMock()
        expander = ContextExpander(retriever)
        result = _make_result(doc_type=DocumentType.SAFARI, text="Google")

        expanded = expander.expand(result)

        assert expanded.context_before == []
        assert expanded.context_after == []


class TestMessageExpansion:
    """Test message expansion via get_thread."""

    def test_message_calls_get_thread(self) -> None:
        """MESSAGE type calls retriever.get_thread with correct chat_id."""
        retriever = MagicMock()
        pivot_ts = datetime(2025, 1, 15, 10, 30)
        result = _make_result(
            doc_type=DocumentType.MESSAGE,
            text="the plumber is coming",
            timestamp=pivot_ts,
            metadata={"chat_id": 42},
            doc_id="pivot",
        )

        before_doc = _make_document(
            text="earlier msg",
            timestamp=pivot_ts - timedelta(minutes=5),
            doc_id="before1",
        )
        after_doc = _make_document(
            text="later msg",
            timestamp=pivot_ts + timedelta(minutes=5),
            doc_id="after1",
        )
        retriever.get_thread.return_value = [before_doc, result.document, after_doc]

        expander = ContextExpander(retriever, window=5)
        expanded = expander.expand(result)

        retriever.get_thread.assert_called_once_with(
            thread_id=42,
            around_timestamp=pivot_ts,
            window=10,
        )
        assert len(expanded.context_before) == 1
        assert expanded.context_before[0].id == "before1"
        assert len(expanded.context_after) == 1
        assert expanded.context_after[0].id == "after1"

    def test_message_missing_chat_id_returns_empty(self) -> None:
        """MESSAGE without chat_id in metadata returns empty context."""
        retriever = MagicMock()
        result = _make_result(
            doc_type=DocumentType.MESSAGE,
            metadata={},
        )

        expander = ContextExpander(retriever)
        expanded = expander.expand(result)

        assert expanded.context_before == []
        assert expanded.context_after == []
        retriever.get_thread.assert_not_called()

    def test_message_splits_before_after_correctly(self) -> None:
        """Thread documents are correctly split into before/after the pivot."""
        retriever = MagicMock()
        pivot_ts = datetime(2025, 1, 15, 12, 0)
        result = _make_result(
            doc_type=DocumentType.MESSAGE,
            timestamp=pivot_ts,
            metadata={"chat_id": 1},
            doc_id="pivot",
        )

        docs = [
            _make_document(text="msg1", timestamp=pivot_ts - timedelta(hours=2), doc_id="d1"),
            _make_document(text="msg2", timestamp=pivot_ts - timedelta(hours=1), doc_id="d2"),
            result.document,
            _make_document(text="msg3", timestamp=pivot_ts + timedelta(hours=1), doc_id="d3"),
        ]
        retriever.get_thread.return_value = docs

        expander = ContextExpander(retriever)
        expanded = expander.expand(result)

        assert len(expanded.context_before) == 2
        assert expanded.context_before[0].id == "d1"
        assert expanded.context_before[1].id == "d2"
        assert len(expanded.context_after) == 1
        assert expanded.context_after[0].id == "d3"


class TestTemporalExpansion:
    """Test temporal expansion for photos and calendar."""

    def test_photo_searches_within_one_hour(self) -> None:
        """PHOTO type searches within ±1 hour window."""
        retriever = MagicMock()
        pivot_ts = datetime(2025, 1, 15, 14, 0)
        result = _make_result(
            doc_type=DocumentType.PHOTO,
            timestamp=pivot_ts,
            doc_id="photo1",
        )

        nearby_photo = _make_document(
            doc_type=DocumentType.PHOTO,
            timestamp=pivot_ts + timedelta(minutes=30),
            doc_id="photo2",
        )
        retriever.search.return_value = [
            Result(document=nearby_photo, score=0.8),
        ]

        expander = ContextExpander(retriever, window=5)
        expanded = expander.expand(result)

        retriever.search.assert_called_once()
        call_args = retriever.search.call_args
        filters = call_args.kwargs.get("filters") or call_args[1].get("filters") or call_args[0][0]
        assert filters.date_after == pivot_ts - timedelta(hours=1)
        assert filters.date_before == pivot_ts + timedelta(hours=1)
        assert len(expanded.context_after) == 1

    def test_calendar_searches_within_one_day(self) -> None:
        """CALENDAR type searches within ±1 day window."""
        retriever = MagicMock()
        pivot_ts = datetime(2025, 1, 15, 14, 0)
        result = _make_result(
            doc_type=DocumentType.CALENDAR,
            timestamp=pivot_ts,
            doc_id="cal1",
        )
        retriever.search.return_value = []

        expander = ContextExpander(retriever)
        expanded = expander.expand(result)

        retriever.search.assert_called_once()
        call_args = retriever.search.call_args
        filters = call_args.kwargs.get("filters") or call_args[1].get("filters") or call_args[0][0]
        assert filters.date_after == pivot_ts - timedelta(days=1)
        assert filters.date_before == pivot_ts + timedelta(days=1)

    def test_photo_missing_timestamp_returns_empty(self) -> None:
        """PHOTO with no timestamp returns empty context."""
        retriever = MagicMock()
        doc = Document(
            type=DocumentType.PHOTO,
            text="",
            source=_make_source(),
            timestamp=None,
        )
        result = Result(document=doc, score=0.9)

        expander = ContextExpander(retriever)
        expanded = expander.expand(result)

        assert expanded.context_before == []
        assert expanded.context_after == []
        retriever.search.assert_not_called()


class TestCallExpansion:
    """Test call/voicemail expansion."""

    def test_call_searches_same_contact(self) -> None:
        """CALL type searches same contact within ±1 day."""
        retriever = MagicMock()
        pivot_ts = datetime(2025, 1, 15, 14, 0)
        result = _make_result(
            doc_type=DocumentType.CALL,
            text="Call with +1555",
            timestamp=pivot_ts,
            metadata={"handle": "+15551234567"},
            doc_id="call1",
        )
        retriever.search.return_value = []

        expander = ContextExpander(retriever)
        expanded = expander.expand(result)

        retriever.search.assert_called_once()
        call_args = retriever.search.call_args
        filters = call_args.kwargs.get("filters") or call_args[1].get("filters") or call_args[0][0]
        assert filters.contacts == ["+15551234567"]
        assert filters.date_after == pivot_ts - timedelta(days=1)
        assert filters.date_before == pivot_ts + timedelta(days=1)

    def test_call_missing_handle_returns_empty(self) -> None:
        """CALL without handle metadata returns empty context."""
        retriever = MagicMock()
        result = _make_result(
            doc_type=DocumentType.CALL,
            metadata={},
        )

        expander = ContextExpander(retriever)
        expanded = expander.expand(result)

        assert expanded.context_before == []
        assert expanded.context_after == []

    def test_call_missing_timestamp_returns_empty(self) -> None:
        """CALL without timestamp returns empty context."""
        retriever = MagicMock()
        doc = Document(
            type=DocumentType.CALL,
            text="Call",
            source=_make_source(),
            timestamp=None,
            metadata={"handle": "+15551234567"},
        )
        result = Result(document=doc, score=0.9)

        expander = ContextExpander(retriever)
        expanded = expander.expand(result)

        assert expanded.context_before == []
        assert expanded.context_after == []


class TestExpandBatch:
    """Test batch expansion with deduplication."""

    def test_batch_deduplicates_shared_context(self) -> None:
        """Shared context documents are not repeated across expanded results."""
        retriever = MagicMock()
        shared_doc = _make_document(
            text="shared context",
            timestamp=datetime(2025, 1, 15, 10, 0),
            doc_id="shared",
        )

        pivot_ts_1 = datetime(2025, 1, 15, 10, 30)
        pivot_ts_2 = datetime(2025, 1, 15, 10, 35)

        result1 = _make_result(
            doc_type=DocumentType.MESSAGE,
            timestamp=pivot_ts_1,
            metadata={"chat_id": 1},
            doc_id="r1",
        )
        result2 = _make_result(
            doc_type=DocumentType.MESSAGE,
            timestamp=pivot_ts_2,
            metadata={"chat_id": 1},
            doc_id="r2",
        )

        # Both threads return the shared doc as context
        retriever.get_thread.side_effect = [
            [shared_doc, result1.document],
            [shared_doc, result2.document],
        ]

        expander = ContextExpander(retriever)
        expanded = expander.expand_batch([result1, result2])

        assert len(expanded) == 2
        # First result should have the shared doc
        all_first_ids = {d.id for d in expanded[0].context_before}
        assert "shared" in all_first_ids
        # Second result should NOT duplicate the shared doc
        all_second_ids = {d.id for d in expanded[1].context_before}
        assert "shared" not in all_second_ids


class TestSplitAround:
    """Test the _split_around static method."""

    def test_splits_correctly(self) -> None:
        """Documents are split into before/after the pivot chronologically."""
        pivot = _make_document(
            timestamp=datetime(2025, 1, 15, 12, 0),
            doc_id="pivot",
        )
        docs = [
            _make_document(text="a", timestamp=datetime(2025, 1, 15, 11, 0), doc_id="a"),
            _make_document(text="b", timestamp=datetime(2025, 1, 15, 11, 30), doc_id="b"),
            _make_document(text="c", timestamp=datetime(2025, 1, 15, 13, 0), doc_id="c"),
        ]

        before, after = ContextExpander._split_around(docs, pivot)

        assert [d.id for d in before] == ["a", "b"]
        assert [d.id for d in after] == ["c"]

    def test_excludes_pivot_from_results(self) -> None:
        """Pivot document is excluded from both before and after."""
        pivot = _make_document(
            timestamp=datetime(2025, 1, 15, 12, 0),
            doc_id="pivot",
        )
        docs = [pivot]

        before, after = ContextExpander._split_around(docs, pivot)

        assert before == []
        assert after == []

    def test_excludes_docs_without_timestamp(self) -> None:
        """Documents without timestamps are excluded."""
        pivot = _make_document(
            timestamp=datetime(2025, 1, 15, 12, 0),
            doc_id="pivot",
        )
        no_ts_doc = Document(
            type=DocumentType.MESSAGE,
            text="no ts",
            source=_make_source(),
            timestamp=None,
        )

        before, after = ContextExpander._split_around([no_ts_doc], pivot)

        assert before == []
        assert after == []

    def test_pivot_without_timestamp_returns_empty(self) -> None:
        """Pivot without timestamp returns empty before/after."""
        pivot = Document(
            type=DocumentType.MESSAGE,
            text="no ts",
            source=_make_source(),
            timestamp=None,
        )
        docs = [
            _make_document(text="a", timestamp=datetime(2025, 1, 15, 11, 0), doc_id="a"),
        ]

        before, after = ContextExpander._split_around(docs, pivot)

        assert before == []
        assert after == []

    def test_before_after_sorted_chronologically(self) -> None:
        """Both before and after lists are sorted oldest-first."""
        pivot = _make_document(
            timestamp=datetime(2025, 1, 15, 12, 0),
            doc_id="pivot",
        )
        docs = [
            _make_document(text="c", timestamp=datetime(2025, 1, 15, 13, 0), doc_id="c"),
            _make_document(text="a", timestamp=datetime(2025, 1, 15, 10, 0), doc_id="a"),
            _make_document(text="d", timestamp=datetime(2025, 1, 15, 14, 0), doc_id="d"),
            _make_document(text="b", timestamp=datetime(2025, 1, 15, 11, 0), doc_id="b"),
        ]

        before, after = ContextExpander._split_around(docs, pivot)

        assert [d.id for d in before] == ["a", "b"]
        assert [d.id for d in after] == ["c", "d"]
