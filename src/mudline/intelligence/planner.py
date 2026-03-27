"""Query planner — decomposes natural language queries into tool calls via LLM.

Uses an agentic loop: the LLM receives the user query and available tools,
decides which tools to call, results are fed back, and the loop continues
until the LLM produces a text response or max_iterations is reached.

Handles temporal reasoning (e.g. "last week"), contact resolution, and
multi-step queries transparently through LLM tool-use capabilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mudline.intelligence.llm import LLMProvider, LLMResponse, Message, ToolCall
from mudline.intelligence.tools import ToolRegistry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a personal data assistant that searches iOS backup data to answer \
the user's questions. You have access to tools for searching messages, notes, \
photos, contacts, call history, calendar events, and Safari browsing history.

Today's date and time: {now}

Guidelines:
- Use the provided tools to find the information needed to answer the question.
- For temporal expressions like "last week", "yesterday", "this month", \
compute the correct date range based on today's date.
- When a query mentions a person by name, use the get_contact tool first if \
you need to resolve their identity, then search with the contact filter.
- For multi-step queries, break them down: resolve contacts, then search \
with appropriate filters.
- When you have gathered enough information, respond with a clear summary \
of your findings. Do not call tools unnecessarily.
- If no results are found, say so clearly.\
"""


@dataclass
class PlanResult:
    """Result of query planning and execution.

    Args:
        query: Original user query.
        tool_calls: All tool calls made during planning.
        tool_results: Mapping of tool_call_id to the results returned.
        llm_response: The LLM's final response (text summary).
    """

    query: str
    tool_calls: list[ToolCall]
    tool_results: dict[str, list[dict[str, Any]]]
    llm_response: LLMResponse


class QueryPlanner:
    """Decomposes natural language queries into tool calls and executes them.

    Runs an agentic loop where the LLM decides which tools to call based on
    the user query and accumulated results, stopping when the LLM produces
    a text-only response or max_iterations is reached.

    Args:
        llm: LLM provider for generating completions.
        tool_registry: Registry of available tools and their executors.
        max_iterations: Maximum number of LLM round-trips with tool calls.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        *,
        max_iterations: int = 3,
    ) -> None:
        self._llm = llm
        self._tool_registry = tool_registry
        self._max_iterations = max_iterations

    async def plan_and_execute(
        self,
        query: str,
        context: list[Message] | None = None,
    ) -> PlanResult:
        """Decompose a query into tool calls, execute them, return results.

        Args:
            query: Natural language question from the user.
            context: Optional conversation history for follow-up queries.

        Returns:
            PlanResult containing all tool calls, their results, and the
            LLM's final text response summarizing findings.
        """
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S %A"),
        )

        messages: list[Message] = [Message(role="system", content=system_prompt)]

        if context:
            messages.extend(context)

        messages.append(Message(role="user", content=query))

        tool_defs = self._tool_registry.get_tool_defs()
        all_tool_calls: list[ToolCall] = []
        all_tool_results: dict[str, list[dict[str, Any]]] = {}

        last_response: LLMResponse | None = None

        for iteration in range(self._max_iterations):
            logger.debug("Planner iteration %d for query: %s", iteration + 1, query)

            response = await self._llm.complete(messages, tools=tool_defs)
            last_response = response

            if not response.tool_calls:
                logger.debug("LLM returned text response, ending loop")
                break

            all_tool_calls.extend(response.tool_calls)

            # If the LLM also produced text alongside tool calls, include it
            if response.content:
                messages.append(Message(role="assistant", content=response.content))

            result_parts: list[str] = []
            for tool_call in response.tool_calls:
                results = self._execute_tool_call(tool_call)
                all_tool_results[tool_call.id] = results
                result_parts.append(
                    f"Tool results for {tool_call.name} (call {tool_call.id}):\n"
                    f"{json.dumps(results, indent=2, default=str)}"
                )

            results_message = "\n\n".join(result_parts)
            messages.append(Message(role="user", content=results_message))
        else:
            # max_iterations exhausted while LLM still wanted tool calls —
            # do one final completion without tools to force a text response
            logger.warning(
                "Max iterations (%d) reached, forcing final response",
                self._max_iterations,
            )
            last_response = await self._llm.complete(messages, tools=None)

        assert last_response is not None
        return PlanResult(
            query=query,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            llm_response=last_response,
        )

    def _execute_tool_call(
        self, tool_call: ToolCall
    ) -> list[dict[str, Any]]:
        """Execute a single tool call, returning results or an error dict.

        Catches exceptions from the tool registry so a single failed tool
        call doesn't abort the entire planning loop.
        """
        try:
            results = self._tool_registry.execute(tool_call.name, tool_call.arguments)
            logger.info(
                "Tool %s (call %s) returned %d results",
                tool_call.name,
                tool_call.id,
                len(results),
            )
            return results
        except Exception as exc:
            logger.error(
                "Tool %s (call %s) failed: %s",
                tool_call.name,
                tool_call.id,
                exc,
            )
            return [{"error": str(exc)}]
