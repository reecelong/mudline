"""Query router — uses a small local LLM to parse queries into structured filters.

A lightweight Ollama model (e.g. qwen3:4b) makes a fast judgment call on each
query: either parse it into structured filters for instant search, or escalate
to the full planner pipeline when reasoning/synthesis is needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from mudline.models.document import DocumentType
from mudline.models.retriever import Filters

if TYPE_CHECKING:
    from mudline.index.contacts import ContactIndex

logger = logging.getLogger(__name__)

_ROUTER_PROMPT = """\
Parse this search query into JSON. Today: {today}
Types: message, photo, note, contact, calendar, call, voicemail, safari
"recent" = date_after {week_ago}. "last month" = date_after {month_ago}.
needs_llm=true ONLY for summarize/explain/analyze/why questions.

"messages from John last week" → {{"type":"message","contact":"John","date_after":"{week_ago}","date_before":"{today}","search_terms":null,"needs_llm":false}}
"summarize chats with Sarah" → {{"type":"message","contact":"Sarah","date_after":null,"date_before":null,"search_terms":null,"needs_llm":true}}
"recent calls" → {{"type":"call","contact":null,"date_after":"{week_ago}","date_before":"{today}","search_terms":null,"needs_llm":false}}
"photos from the beach" → {{"type":"photo","contact":null,"date_after":null,"date_before":null,"search_terms":"beach","needs_llm":false}}

"{query}" →"""


@dataclass
class RouteResult:
    """Result of the query routing decision."""

    needs_llm: bool
    filters: Filters | None = None
    search_query: str | None = None
    reason: str = ""
    raw_parse: dict[str, Any] = field(default_factory=dict)


class QueryRouter:
    """Routes queries using a small local LLM for intent parsing.

    Sends the query to a fast local model (via Ollama) which parses it into
    structured filters. If the model determines the query needs reasoning
    or synthesis, it flags it for escalation to the full planner pipeline.

    Args:
        contact_index: Contact index for resolving names to handles.
        ollama_url: Base URL for the Ollama API.
        model: Ollama model to use for routing (should be small/fast).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        contact_index: ContactIndex,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3.5:4b",
        timeout: float = 10.0,
    ) -> None:
        self._contact_index = contact_index
        self._ollama_url = ollama_url
        self._model = model
        self._timeout = timeout

    def route(self, query: str) -> RouteResult:
        """Parse a query into structured filters using the local LLM.

        Falls back to escalation if the local model is unavailable or
        returns unparseable output.

        Args:
            query: Natural language query from the user.

        Returns:
            RouteResult with parsed filters or escalation flag.
        """
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        week_ago = (now - timedelta(weeks=1)).strftime("%Y-%m-%d")
        month_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d")

        prompt = _ROUTER_PROMPT.format(
            today=today,
            week_ago=week_ago,
            month_ago=month_ago,
            query=query,
        )

        try:
            parsed = self._call_ollama(prompt)
        except Exception as e:
            logger.warning("Router LLM failed, escalating: %s", e)
            return RouteResult(
                needs_llm=True,
                reason=f"router model unavailable: {e}",
            )

        if parsed.get("needs_llm", True):
            return RouteResult(
                needs_llm=True,
                reason="model determined query needs reasoning",
                raw_parse=parsed,
            )

        # Build filters from parsed output
        filters = self._build_filters(parsed)
        search_query = parsed.get("search_terms")

        if not filters and not search_query:
            return RouteResult(
                needs_llm=True,
                reason="no usable filters parsed",
                raw_parse=parsed,
            )

        logger.info("Fast path: %s", parsed)

        return RouteResult(
            needs_llm=False,
            filters=filters,
            search_query=search_query,
            reason="parsed into structured filters",
            raw_parse=parsed,
        )

    def _call_ollama(self, prompt: str) -> dict[str, Any]:
        """Call the local Ollama model and parse JSON from its response."""
        response = httpx.post(
            f"{self._ollama_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 300,
                },
            },
            timeout=self._timeout,
        )
        response.raise_for_status()

        text = response.json().get("response", "")
        logger.debug("Router model response: %s", text)

        # Extract JSON from response (model may include extra text)
        return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract a JSON object from model output, handling markdown fences."""
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from model output: {text[:200]}")

    def _build_filters(self, parsed: dict[str, Any]) -> Filters | None:
        """Build a Filters object from the parsed model output."""
        doc_type = None
        type_str = parsed.get("type")
        if type_str:
            try:
                doc_type = DocumentType(type_str)
            except ValueError:
                logger.warning("Unknown document type from router: %s", type_str)

        # Resolve contact name to handles
        contacts = None
        contact_name = parsed.get("contact")
        if contact_name:
            handles = self._contact_index.resolve(contact_name)
            if handles:
                contacts = handles
            else:
                # Use the raw name as a fallback handle
                contacts = [contact_name]

        # Parse dates
        date_after = self._parse_date(parsed.get("date_after"))
        date_before = self._parse_date(parsed.get("date_before"), is_upper_bound=True)

        has_anything = any([doc_type, contacts, date_after, date_before])
        if not has_anything:
            return None

        return Filters(
            data_types=[doc_type] if doc_type else None,
            contacts=contacts,
            date_after=date_after,
            date_before=date_before,
        )

    @staticmethod
    def _parse_date(value: Any, *, is_upper_bound: bool = False) -> datetime | None:
        """Parse an ISO date string from model output.

        Args:
            value: Date string from the model.
            is_upper_bound: If True and the value is a bare date (no time),
                           set time to end of day instead of midnight.
        """
        if not value or not isinstance(value, str):
            return None
        try:
            dt = datetime.fromisoformat(value)
            # If it's a bare date (midnight) used as an upper bound,
            # extend to end of day so "today" includes all of today
            if is_upper_bound and dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt
        except ValueError:
            logger.warning("Could not parse date from router: %s", value)
            return None
