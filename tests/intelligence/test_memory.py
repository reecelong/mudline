"""Unit tests for ConversationMemory — sliding-window conversation state.

Tests verify:
- Adding user and assistant messages
- get_context returns messages in order
- max_turns windowing
- Leading assistant message trimming
- clear() empties all messages
- turn_count counting logic
- Window enforcement on overflow
"""

from __future__ import annotations

import pytest

from mudline.intelligence.memory import ConversationMemory


class TestAddMessages:
    """Test adding messages to memory."""

    def test_add_user_message(self) -> None:
        """add_user_message adds a message with role='user'."""
        mem = ConversationMemory()
        mem.add_user_message("hello")

        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0].role == "user"
        assert ctx[0].content == "hello"

    def test_add_assistant_message(self) -> None:
        """add_assistant_message adds a message with role='assistant'."""
        mem = ConversationMemory()
        mem.add_user_message("hi")
        mem.add_assistant_message("hello back")

        ctx = mem.get_context()
        assert len(ctx) == 2
        assert ctx[1].role == "assistant"
        assert ctx[1].content == "hello back"


class TestGetContext:
    """Test context retrieval behavior."""

    def test_returns_messages_in_order(self) -> None:
        """Messages are returned in the order they were added."""
        mem = ConversationMemory()
        mem.add_user_message("first")
        mem.add_assistant_message("second")
        mem.add_user_message("third")

        ctx = mem.get_context()
        assert [m.content for m in ctx] == ["first", "second", "third"]

    def test_respects_max_turns_window(self) -> None:
        """Only the most recent max_turns worth of messages are returned."""
        mem = ConversationMemory(max_turns=2)

        mem.add_user_message("u1")
        mem.add_assistant_message("a1")
        mem.add_user_message("u2")
        mem.add_assistant_message("a2")
        mem.add_user_message("u3")
        mem.add_assistant_message("a3")

        ctx = mem.get_context()
        # max_turns=2 means 4 messages max
        assert len(ctx) <= 4
        # Should contain the most recent turns
        contents = [m.content for m in ctx]
        assert "u3" in contents
        assert "a3" in contents

    def test_trims_leading_assistant_messages(self) -> None:
        """Context always starts with a user message, trimming leading assistant."""
        mem = ConversationMemory(max_turns=1)

        # Add enough to push window so it starts with assistant
        mem.add_user_message("u1")
        mem.add_assistant_message("a1")
        mem.add_user_message("u2")
        mem.add_assistant_message("a2")
        mem.add_user_message("u3")

        ctx = mem.get_context()
        if ctx:
            assert ctx[0].role == "user"

    def test_empty_memory_returns_empty(self) -> None:
        """Empty memory returns empty context."""
        mem = ConversationMemory()
        assert mem.get_context() == []


class TestClear:
    """Test clearing conversation history."""

    def test_clear_empties_all_messages(self) -> None:
        """clear() removes all stored messages."""
        mem = ConversationMemory()
        mem.add_user_message("hello")
        mem.add_assistant_message("hi")

        mem.clear()

        assert mem.get_context() == []
        assert mem.turn_count == 0


class TestTurnCount:
    """Test turn counting logic."""

    def test_complete_pairs(self) -> None:
        """turn_count counts complete user+assistant pairs."""
        mem = ConversationMemory()
        mem.add_user_message("u1")
        mem.add_assistant_message("a1")
        mem.add_user_message("u2")
        mem.add_assistant_message("a2")

        assert mem.turn_count == 2

    def test_only_user_messages_returns_zero(self) -> None:
        """turn_count with only user messages returns 0."""
        mem = ConversationMemory()
        mem.add_user_message("u1")
        mem.add_user_message("u2")

        assert mem.turn_count == 0

    def test_trailing_user_message_not_counted(self) -> None:
        """Incomplete turn (user without following assistant) is not counted."""
        mem = ConversationMemory()
        mem.add_user_message("u1")
        mem.add_assistant_message("a1")
        mem.add_user_message("u2")

        assert mem.turn_count == 1

    def test_empty_memory_returns_zero(self) -> None:
        """Empty memory has zero turns."""
        mem = ConversationMemory()
        assert mem.turn_count == 0


class TestWindowEnforcement:
    """Test sliding window enforcement on overflow."""

    def test_adding_beyond_max_trims_oldest(self) -> None:
        """Adding beyond max_turns trims oldest messages."""
        mem = ConversationMemory(max_turns=2)

        for i in range(5):
            mem.add_user_message(f"u{i}")
            mem.add_assistant_message(f"a{i}")

        ctx = mem.get_context()
        # Should only keep last 2 turns = 4 messages max
        assert len(ctx) <= 4

        # Oldest messages should be gone
        contents = [m.content for m in ctx]
        assert "u0" not in contents
        assert "a0" not in contents

    def test_window_always_starts_with_user(self) -> None:
        """After window enforcement, context always starts with user message."""
        mem = ConversationMemory(max_turns=1)

        mem.add_user_message("u1")
        mem.add_assistant_message("a1")
        mem.add_user_message("u2")
        mem.add_assistant_message("a2")

        ctx = mem.get_context()
        if ctx:
            assert ctx[0].role == "user"
