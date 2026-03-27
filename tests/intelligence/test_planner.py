"""Unit tests for QueryPlanner — query decomposition and agentic tool execution.

Tests verify:
- Text-only responses end the loop immediately
- Tool calls are executed and results fed back
- Max iterations limit is respected
- Context messages are included in conversation
- Tool execution error handling
- System prompt construction
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mudline.intelligence.llm import LLMResponse, Message, ToolCall, ToolDef
from mudline.intelligence.planner import PlanResult, QueryPlanner


def _make_tool_defs() -> list[ToolDef]:
    return [
        ToolDef(
            name="search_messages",
            description="Search text messages",
            parameters={"type": "object", "properties": {}},
        ),
    ]


def _text_response(content: str = "Here is your answer.") -> LLMResponse:
    return LLMResponse(content=content, model="test-model")


def _tool_response(
    tool_calls: list[ToolCall],
    content: str = "",
) -> LLMResponse:
    return LLMResponse(content=content, model="test-model", tool_calls=tool_calls)


class TestTextOnlyResponse:
    """Test behavior when LLM returns text without tool calls."""

    @pytest.mark.asyncio
    async def test_text_response_returns_immediately(self) -> None:
        """LLM returning text (no tool calls) ends the loop with empty tool_calls."""
        llm = AsyncMock()
        llm.complete.return_value = _text_response("No results found.")

        tool_registry = MagicMock()
        tool_registry.get_tool_defs.return_value = _make_tool_defs()

        planner = QueryPlanner(llm=llm, tool_registry=tool_registry)
        result = await planner.plan_and_execute("what did Sarah say?")

        assert isinstance(result, PlanResult)
        assert result.query == "what did Sarah say?"
        assert result.tool_calls == []
        assert result.tool_results == {}
        assert result.llm_response.content == "No results found."
        llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_prompt_includes_datetime(self) -> None:
        """System prompt includes current date/time."""
        llm = AsyncMock()
        llm.complete.return_value = _text_response()

        tool_registry = MagicMock()
        tool_registry.get_tool_defs.return_value = _make_tool_defs()

        planner = QueryPlanner(llm=llm, tool_registry=tool_registry)
        await planner.plan_and_execute("test")

        call_args = llm.complete.call_args
        messages = call_args[0][0]
        system_msg = messages[0]
        assert system_msg.role == "system"
        assert "Today's date and time:" in system_msg.content


class TestToolCallExecution:
    """Test tool call execution and feedback loop."""

    @pytest.mark.asyncio
    async def test_tool_calls_executed_and_fed_back(self) -> None:
        """LLM tool calls are executed, results fed back, then LLM returns text."""
        tool_call = ToolCall(id="tc1", name="search_messages", arguments={"query": "plumber"})

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[
            _tool_response([tool_call]),
            _text_response("Found 3 messages about plumber."),
        ])

        tool_registry = MagicMock()
        tool_registry.get_tool_defs.return_value = _make_tool_defs()
        tool_registry.execute.return_value = [
            {"id": "abc", "text": "the plumber is coming", "type": "message"}
        ]

        planner = QueryPlanner(llm=llm, tool_registry=tool_registry)
        result = await planner.plan_and_execute("what about the plumber?")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_messages"
        assert "tc1" in result.tool_results
        assert result.tool_results["tc1"] == [
            {"id": "abc", "text": "the plumber is coming", "type": "message"}
        ]
        assert result.llm_response.content == "Found 3 messages about plumber."
        tool_registry.execute.assert_called_once_with("search_messages", {"query": "plumber"})

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_response(self) -> None:
        """Multiple tool calls in one LLM response are all executed."""
        tc1 = ToolCall(id="tc1", name="search_messages", arguments={"query": "a"})
        tc2 = ToolCall(id="tc2", name="search_messages", arguments={"query": "b"})

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[
            _tool_response([tc1, tc2]),
            _text_response("Done."),
        ])

        tool_registry = MagicMock()
        tool_registry.get_tool_defs.return_value = _make_tool_defs()
        tool_registry.execute.return_value = []

        planner = QueryPlanner(llm=llm, tool_registry=tool_registry)
        result = await planner.plan_and_execute("search two things")

        assert len(result.tool_calls) == 2
        assert tool_registry.execute.call_count == 2


class TestMaxIterations:
    """Test iteration limit enforcement."""

    @pytest.mark.asyncio
    async def test_max_iterations_forces_final_response(self) -> None:
        """When max_iterations is exhausted, a final LLM call without tools is made."""
        tool_call = ToolCall(id="tc", name="search_messages", arguments={"query": "x"})

        llm = AsyncMock()
        # Return tool calls for max_iterations, then final text on forced call
        llm.complete = AsyncMock(side_effect=[
            _tool_response([tool_call]),
            _tool_response([tool_call]),
            _text_response("Forced final answer."),
        ])

        tool_registry = MagicMock()
        tool_registry.get_tool_defs.return_value = _make_tool_defs()
        tool_registry.execute.return_value = []

        planner = QueryPlanner(llm=llm, tool_registry=tool_registry, max_iterations=2)
        result = await planner.plan_and_execute("complex query")

        assert result.llm_response.content == "Forced final answer."
        # 2 iterations + 1 forced final = 3 calls
        assert llm.complete.call_count == 3
        # The forced final call should have tools=None
        last_call = llm.complete.call_args_list[-1]
        assert last_call.kwargs.get("tools") is None or (
            len(last_call[0]) > 1 and last_call[0][1] is None
        )


class TestContextMessages:
    """Test that conversation context is included."""

    @pytest.mark.asyncio
    async def test_context_messages_included(self) -> None:
        """Context messages are inserted between system prompt and user query."""
        llm = AsyncMock()
        llm.complete.return_value = _text_response("answer")

        tool_registry = MagicMock()
        tool_registry.get_tool_defs.return_value = _make_tool_defs()

        context = [
            Message(role="user", content="Who is Sarah?"),
            Message(role="assistant", content="Sarah is your friend."),
        ]

        planner = QueryPlanner(llm=llm, tool_registry=tool_registry)
        await planner.plan_and_execute("what did she say yesterday?", context=context)

        call_args = llm.complete.call_args
        messages = call_args[0][0]
        # system + 2 context + user query = 4
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == "Who is Sarah?"
        assert messages[2].role == "assistant"
        assert messages[3].role == "user"
        assert messages[3].content == "what did she say yesterday?"


class TestToolExecutionErrorHandling:
    """Test error handling during tool execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_call_catches_exceptions(self) -> None:
        """Tool execution exceptions are caught and returned as error dicts."""
        tool_call = ToolCall(id="tc1", name="search_messages", arguments={"query": "x"})

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[
            _tool_response([tool_call]),
            _text_response("Could not find results."),
        ])

        tool_registry = MagicMock()
        tool_registry.get_tool_defs.return_value = _make_tool_defs()
        tool_registry.execute.side_effect = RuntimeError("Database connection failed")

        planner = QueryPlanner(llm=llm, tool_registry=tool_registry)
        result = await planner.plan_and_execute("search something")

        assert "tc1" in result.tool_results
        assert result.tool_results["tc1"] == [{"error": "Database connection failed"}]
