"""Q-04: Synthesizer — produces natural language answers with citations from search results.

Takes retrieved results (with expanded context from ContextExpander) and uses
the LLM to generate conversational answers that reference specific messages,
dates, contacts, and other source material.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mudline.intelligence.llm import LLMResponse, Message
from mudline.models.document import Document, DocumentType

if TYPE_CHECKING:
    from mudline.intelligence.context import ExpandedResult
    from mudline.intelligence.llm import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a personal assistant answering questions about the user's iOS data \
(messages, notes, photos, calls, calendar events, etc.).

Rules:
- Answer the user's question based ONLY on the provided search results.
- Be conversational but accurate. Do not invent information.
- Cite sources inline using the format [Type from Contact, YYYY-MM-DD] — for example \
[Message from John, 2025-01-15] or [Note, 2025-03-10].
- When multiple conversations or sources are relevant, organize your answer clearly \
("Here are 3 relevant conversations...").
- If the results don't contain enough information to answer, say so honestly.
- For messages, preserve the conversational tone and quote key phrases when helpful.
- Keep answers concise unless the user's question calls for detail."""


@dataclass(frozen=True)
class Citation:
    """A reference to a specific source document.

    Args:
        document_id: The unique ID of the referenced document.
        document_type: Human-readable type ("message", "note", etc.).
        timestamp: ISO-formatted timestamp, or None if unavailable.
        contact: Contact name or handle, if applicable.
        excerpt: Short excerpt from the source document.
    """

    document_id: str
    document_type: str
    timestamp: str | None
    contact: str | None
    excerpt: str


@dataclass
class SynthesizedAnswer:
    """A natural language answer with citations.

    Args:
        text: The generated answer text with inline citations.
        citations: Source references for all evidence documents.
        usage: LLM token usage information.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


class Synthesizer:
    """Synthesizes natural language answers from search results using an LLM.

    Args:
        llm: The LLM provider to use for generating answers.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def synthesize(
        self,
        query: str,
        plan_result: Any,
        expanded_results: list[ExpandedResult] | None = None,
    ) -> SynthesizedAnswer:
        """Synthesize a natural language answer from search results.

        Formats the evidence from plan_result and expanded_results into a prompt,
        sends it to the LLM, and extracts citations from the referenced documents.

        Args:
            query: The original user question.
            plan_result: PlanResult from the query planner containing tool_results.
            expanded_results: Optional context-expanded results for richer threads.

        Returns:
            SynthesizedAnswer with text, citations, and usage info.
        """
        doc_lookup = self._build_document_lookup(plan_result, expanded_results)
        evidence = self._format_evidence(plan_result, expanded_results)

        if not evidence.strip():
            return SynthesizedAnswer(
                text="I couldn't find any relevant information to answer your question.",
            )

        user_content = f"Question: {query}\n\n--- Search Results ---\n{evidence}"

        response: LLMResponse = await self._llm.complete(
            messages=[Message(role="user", content=_SYSTEM_PROMPT + "\n\n" + user_content)],
        )

        citations = self._build_citations(doc_lookup)

        logger.debug(
            "Synthesized answer: %d chars, %d citations, model=%s",
            len(response.content),
            len(citations),
            response.model,
        )

        return SynthesizedAnswer(
            text=response.content,
            citations=citations,
            usage=response.usage,
        )

    def _build_document_lookup(
        self,
        plan_result: Any,
        expanded_results: list[ExpandedResult] | None,
    ) -> dict[str, Document | dict[str, Any]]:
        """Build a lookup of all evidence by document ID.

        Tool results are serialized dicts (from ToolRegistry.execute), while
        expanded results contain actual Document objects.
        """
        lookup: dict[str, Document | dict[str, Any]] = {}

        tool_results: dict[str, list[dict[str, Any]]] = getattr(plan_result, "tool_results", {})
        for results_list in tool_results.values():
            for item in results_list:
                doc_id = item.get("id")
                if doc_id:
                    lookup[doc_id] = item

        if expanded_results:
            for er in expanded_results:
                doc = er.result.document
                if doc.id:
                    lookup[doc.id] = doc
                for ctx_doc in er.context_before:
                    if ctx_doc.id:
                        lookup[ctx_doc.id] = ctx_doc
                for ctx_doc in er.context_after:
                    if ctx_doc.id:
                        lookup[ctx_doc.id] = ctx_doc

        return lookup

    def _format_evidence(
        self,
        plan_result: Any,
        expanded_results: list[ExpandedResult] | None,
    ) -> str:
        """Format all evidence into a text block for the LLM prompt."""
        sections: list[str] = []

        # Format tool results from the planner (these are serialized dicts)
        tool_results: dict[str, list[dict[str, Any]]] = getattr(plan_result, "tool_results", {})
        for _call_id, results_list in tool_results.items():
            if not results_list:
                continue
            tool_lines: list[str] = []
            for item in results_list:
                tool_lines.append(_format_result_dict(item))
            if tool_lines:
                sections.append("\n".join(tool_lines))

        # Format expanded results with conversation context
        if expanded_results:
            for er in expanded_results:
                sections.append(_format_expanded_result(er))

        return "\n\n".join(sections)

    def _build_citations(
        self, doc_lookup: dict[str, Document | dict[str, Any]]
    ) -> list[Citation]:
        """Build Citation objects for all evidence documents.

        Handles both Document objects (from expanded results) and serialized
        dicts (from tool results).
        """
        citations: list[Citation] = []
        for doc_id, entry in doc_lookup.items():
            if isinstance(entry, Document):
                contact = _extract_contact(entry)
                timestamp_str = entry.timestamp.isoformat() if entry.timestamp else None
                excerpt = entry.text[:120].strip() if entry.text else ""
                doc_type = entry.type.value
            else:
                contact = entry.get("metadata", {}).get("handle")
                timestamp_str = entry.get("timestamp")
                excerpt = (entry.get("text") or "")[:120].strip()
                doc_type = entry.get("type", "unknown")

            citations.append(
                Citation(
                    document_id=doc_id,
                    document_type=doc_type,
                    timestamp=timestamp_str,
                    contact=contact,
                    excerpt=excerpt,
                )
            )
        return citations


def _format_result_dict(item: dict[str, Any]) -> str:
    """Format a serialized result dict (from ToolRegistry) as a readable line."""
    ts = item.get("timestamp", "unknown date")
    text = item.get("text", "")
    doc_type = item.get("type", "")
    meta = item.get("metadata", {})

    if doc_type == "message":
        direction = "From" if not meta.get("is_from_me") else "To"
        handle = meta.get("handle", "unknown")
        return f"[{ts}] {direction} {handle}: {text}"
    if doc_type == "call":
        call_type = meta.get("call_type", "call")
        handle = meta.get("handle", "unknown")
        duration = meta.get("duration_seconds", 0)
        mins, secs = divmod(duration, 60)
        return f"[{ts}] {call_type.capitalize()} call with {handle} ({mins}m {secs}s)"
    if doc_type == "note":
        folder = meta.get("folder", "")
        label = f"Note ({folder})" if folder else "Note"
        return f"{label}: {text}"
    if doc_type == "contact":
        return f"Contact: {text}"
    if doc_type == "photo":
        parts = [f"[{ts}] Photo"]
        if meta.get("album"):
            parts.append(f'in album "{meta["album"]}"')
        return " ".join(parts)
    if doc_type == "calendar":
        location = meta.get("location")
        parts = [f"[{ts}] Event: {text}"]
        if location:
            parts.append(f"at {location}")
        return " — ".join(parts)
    if doc_type == "safari":
        url = meta.get("url", "")
        return f"[{ts}] Safari: {text} ({url})"
    return f"[{ts}] {doc_type}: {text}"


def _extract_contact(doc: Document) -> str | None:
    """Extract the most useful contact identifier from a document's metadata."""
    meta = doc.metadata
    if doc.type == DocumentType.MESSAGE:
        return meta.get("handle")
    if doc.type in (DocumentType.CALL, DocumentType.VOICEMAIL):
        return meta.get("handle")
    if doc.type == DocumentType.CALENDAR:
        attendees = meta.get("attendees")
        if attendees and isinstance(attendees, list):
            return ", ".join(attendees[:3])
    return None


def _format_document(doc: Document) -> str:
    """Format a single document as a human-readable line for the LLM.

    Dispatches to type-specific formatting.
    """
    ts = doc.timestamp.strftime("%Y-%m-%d %H:%M") if doc.timestamp else "unknown date"
    meta = doc.metadata

    match doc.type:
        case DocumentType.MESSAGE:
            direction = "From" if not meta.get("is_from_me") else "To"
            handle = meta.get("handle", "unknown")
            return f"[{ts}] {direction} {handle}: {doc.text}"

        case DocumentType.NOTE:
            folder = meta.get("folder", "")
            label = f"Note ({folder})" if folder else "Note"
            return f"{label}: {doc.text}"

        case DocumentType.PHOTO:
            parts = [f"[{ts}] Photo"]
            if meta.get("latitude") is not None:
                parts.append(f"at ({meta['latitude']:.4f}, {meta['longitude']:.4f})")
            if meta.get("album"):
                parts.append(f'in album "{meta["album"]}"')
            if doc.text:
                parts.append(f"— {doc.text}")
            return " ".join(parts)

        case DocumentType.CALL:
            call_type = meta.get("call_type", "call")
            handle = meta.get("handle", "unknown")
            duration = meta.get("duration_seconds", 0)
            mins, secs = divmod(duration, 60)
            return f"[{ts}] {call_type.capitalize()} call with {handle} ({mins}m {secs}s)"

        case DocumentType.VOICEMAIL:
            handle = meta.get("handle", "unknown")
            duration = meta.get("duration_seconds", 0)
            mins, secs = divmod(duration, 60)
            transcription = meta.get("transcription", doc.text)
            return f"[{ts}] Voicemail from {handle} ({mins}m {secs}s): {transcription}"

        case DocumentType.CALENDAR:
            title = doc.text
            location = meta.get("location")
            parts = [f"[{ts}] Event: {title}"]
            if location:
                parts.append(f"at {location}")
            return " — ".join(parts)

        case DocumentType.CONTACT:
            return f"Contact: {doc.text}"

        case DocumentType.SAFARI:
            url = meta.get("url", "")
            return f"[{ts}] Safari: {doc.text} ({url})"

        case _:
            return f"[{ts}] {doc.type.value}: {doc.text}"


def _format_expanded_result(er: ExpandedResult) -> str:
    """Format an ExpandedResult as a conversation thread with context markers."""
    lines: list[str] = []
    has_context = er.context_before or er.context_after

    if has_context:
        lines.append("--- Conversation context ---")

    for doc in er.context_before:
        lines.append(_format_document(doc))

    lines.append(f">>> {_format_document(er.result.document)} <<<")

    for doc in er.context_after:
        lines.append(_format_document(doc))

    return "\n".join(lines)
