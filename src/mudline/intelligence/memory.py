"""Conversation memory for multi-turn interactions.

Manages sliding-window conversation state so follow-up queries retain
context from previous turns (e.g., "What about last Tuesday?" after asking
about John retains the John filter).
"""

from __future__ import annotations

import logging

from mudline.intelligence.llm import Message

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Sliding-window conversation memory for multi-turn interactions.

    Stores user/assistant message pairs and provides them as context
    to the query planner for follow-up queries.

    Args:
        max_turns: Maximum number of conversation turns to retain.
                   A turn is one user message + one assistant response.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._messages: list[Message] = []
        self._max_turns = max_turns

    def add_user_message(self, content: str) -> None:
        """Record a user message."""
        self._messages.append(Message(role="user", content=content))
        self._enforce_window()

    def add_assistant_message(self, content: str) -> None:
        """Record an assistant response."""
        self._messages.append(Message(role="assistant", content=content))
        self._enforce_window()

    def get_context(self) -> list[Message]:
        """Return conversation history for the planner.

        Returns the most recent ``max_turns`` worth of messages,
        always starting with a user message (never an orphaned assistant).
        """
        max_messages = self._max_turns * 2
        window = self._messages[-max_messages:]

        # Trim leading assistant messages so context always starts with user.
        while window and window[0].role != "user":
            window = window[1:]

        return list(window)

    def clear(self) -> None:
        """Clear all conversation history."""
        self._messages.clear()
        logger.debug("Conversation memory cleared")

    @property
    def turn_count(self) -> int:
        """Number of complete turns (user+assistant pairs) stored."""
        pairs = 0
        expecting_assistant = False
        for msg in self._messages:
            if msg.role == "user":
                expecting_assistant = True
            elif msg.role == "assistant" and expecting_assistant:
                pairs += 1
                expecting_assistant = False
        return pairs

    def _enforce_window(self) -> None:
        """Trim messages to stay within the sliding window.

        Keeps at most ``max_turns * 2`` messages. After trimming, drops
        any leading assistant message so context always starts with a
        user message.
        """
        max_messages = self._max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
            # Ensure we don't start with an orphaned assistant message.
            while self._messages and self._messages[0].role != "user":
                self._messages.pop(0)
