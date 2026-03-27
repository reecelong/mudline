"""Q-03: Context Expansion — enriches search results with surrounding context.

Given a search result, expands it with chronologically adjacent documents so
the LLM has enough context to synthesize accurate, cited answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from mudline.models.document import DocumentType

if TYPE_CHECKING:
    from mudline.models.document import Document
    from mudline.models.retriever import Filters, Result, Retriever

logger = logging.getLogger(__name__)

# Document types that are self-contained and need no expansion.
_NO_EXPANSION_TYPES = frozenset({
    DocumentType.NOTE,
    DocumentType.CONTACT,
    DocumentType.SAFARI,
})


@dataclass
class ExpandedResult:
    """A search result with surrounding context documents."""

    result: Result
    context_before: list[Document] = field(default_factory=list)
    context_after: list[Document] = field(default_factory=list)

    @property
    def all_document_ids(self) -> set[str]:
        """All document IDs in this expanded result (original + context)."""
        ids = {self.result.document.id}
        ids.update(d.id for d in self.context_before)
        ids.update(d.id for d in self.context_after)
        return ids


class ContextExpander:
    """Expands search results with surrounding context for LLM consumption.

    Args:
        retriever: The retriever to use for fetching context documents.
        window: Number of context documents to fetch on each side.
    """

    def __init__(self, retriever: Retriever, window: int = 5) -> None:
        self._retriever = retriever
        self._window = window

    def expand(self, result: Result) -> ExpandedResult:
        """Expand a single result with surrounding context.

        Dispatches to type-specific expansion logic based on the document's
        DocumentType. Types without meaningful surrounding context (notes,
        contacts, safari) return an empty ExpandedResult.
        """
        doc = result.document
        doc_type = doc.type

        if doc_type in _NO_EXPANSION_TYPES:
            return ExpandedResult(result=result)

        if doc_type == DocumentType.MESSAGE:
            return self._expand_message(result)
        if doc_type == DocumentType.PHOTO:
            return self._expand_temporal(result, timedelta(hours=1), [DocumentType.PHOTO])
        if doc_type == DocumentType.CALENDAR:
            return self._expand_temporal(result, timedelta(days=1), [DocumentType.CALENDAR])
        if doc_type in (DocumentType.CALL, DocumentType.VOICEMAIL):
            return self._expand_call_voicemail(result)

        logger.warning("No expansion strategy for type %s, returning bare result", doc_type)
        return ExpandedResult(result=result)

    def expand_batch(self, results: list[Result]) -> list[ExpandedResult]:
        """Expand multiple results, deduplicating shared context.

        Documents that already appeared as context (or as the primary result)
        in a previously expanded result are excluded from subsequent context
        lists to avoid redundancy in the LLM prompt.
        """
        seen_ids: set[str] = set()
        expanded: list[ExpandedResult] = []

        for result in results:
            er = self.expand(result)
            er.context_before = [d for d in er.context_before if d.id not in seen_ids]
            er.context_after = [d for d in er.context_after if d.id not in seen_ids]
            seen_ids.update(er.all_document_ids)
            expanded.append(er)

        return expanded

    def _expand_message(self, result: Result) -> ExpandedResult:
        """Expand a message result with surrounding thread context."""
        doc = result.document
        chat_id = doc.metadata.get("chat_id")

        if chat_id is None:
            logger.debug("Message %s has no chat_id, skipping expansion", doc.id)
            return ExpandedResult(result=result)

        thread_docs = self._retriever.get_thread(
            thread_id=int(chat_id),
            around_timestamp=doc.timestamp,
            window=self._window * 2,
        )

        before, after = self._split_around(thread_docs, doc)
        return ExpandedResult(result=result, context_before=before, context_after=after)

    def _expand_temporal(
        self,
        result: Result,
        delta: timedelta,
        data_types: list[DocumentType],
    ) -> ExpandedResult:
        """Expand a result by searching for temporally adjacent documents."""
        from mudline.models.retriever import Filters

        doc = result.document
        if doc.timestamp is None:
            return ExpandedResult(result=result)

        filters = Filters(
            data_types=data_types,
            date_after=doc.timestamp - delta,
            date_before=doc.timestamp + delta,
            backup_id=doc.source.backup_id,
        )

        nearby = self._retriever.search(filters=filters, limit=self._window * 2 + 1)
        nearby_docs = [r.document for r in nearby if r.document.id != doc.id]
        before, after = self._split_around(nearby_docs, doc)
        return ExpandedResult(result=result, context_before=before, context_after=after)

    def _expand_call_voicemail(self, result: Result) -> ExpandedResult:
        """Expand call/voicemail by finding same-contact calls within ±1 day."""
        from mudline.models.retriever import Filters

        doc = result.document
        handle = doc.metadata.get("handle")

        if doc.timestamp is None or handle is None:
            return ExpandedResult(result=result)

        delta = timedelta(days=1)
        filters = Filters(
            data_types=[DocumentType.CALL, DocumentType.VOICEMAIL],
            contacts=[handle],
            date_after=doc.timestamp - delta,
            date_before=doc.timestamp + delta,
            backup_id=doc.source.backup_id,
        )

        nearby = self._retriever.search(filters=filters, limit=self._window * 2 + 1)
        nearby_docs = [r.document for r in nearby if r.document.id != doc.id]
        before, after = self._split_around(nearby_docs, doc)
        return ExpandedResult(result=result, context_before=before, context_after=after)

    @staticmethod
    def _split_around(
        docs: list[Document],
        pivot: Document,
    ) -> tuple[list[Document], list[Document]]:
        """Split a chronological list of documents into before/after the pivot.

        Documents without timestamps are excluded. The returned lists are
        sorted chronologically (oldest first).
        """
        pivot_ts = pivot.timestamp
        if pivot_ts is None:
            return [], []

        before: list[Document] = []
        after: list[Document] = []

        for d in docs:
            if d.id == pivot.id or d.timestamp is None:
                continue
            if d.timestamp <= pivot_ts:
                before.append(d)
            else:
                after.append(d)

        before.sort(key=lambda d: d.timestamp)  # type: ignore[arg-type]
        after.sort(key=lambda d: d.timestamp)  # type: ignore[arg-type]
        return before, after
