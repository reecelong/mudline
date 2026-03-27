"""Tests for MessageExtractor."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from mudline.extractors.messages import MessageExtractor
from mudline.foundation.manifest import ManifestResolver
from mudline.models.document import DocumentType

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def extractor() -> MessageExtractor:
    """Create a MessageExtractor instance."""
    return MessageExtractor()


@pytest.fixture
def resolver(backup_path: Path) -> ManifestResolver:
    """Create a ManifestResolver for the test backup."""
    return ManifestResolver(backup_path)


class TestMessageExtractor:
    """Test suite for MessageExtractor."""

    def test_domain_property(self, extractor: MessageExtractor) -> None:
        """Test that domain property is correct."""
        assert extractor.domain == "HomeDomain"

    def test_data_type_property(self, extractor: MessageExtractor) -> None:
        """Test that data_type property is correct."""
        assert extractor.data_type == "message"

    def test_can_extract_with_sms_db(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that can_extract returns True when sms.db exists."""
        assert extractor.can_extract(resolver) is True

    def test_extract_all_messages(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test extracting all messages from the test fixture."""
        docs = list(extractor.extract(resolver))

        # Fixture has 6 messages
        assert len(docs) == 6

        # All should be MESSAGE type
        assert all(doc.type == DocumentType.MESSAGE for doc in docs)

        # All should have text
        assert all(doc.text for doc in docs)

        # All should have timestamps
        assert all(doc.timestamp is not None for doc in docs)

    def test_message_text_content(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that message text is correctly extracted."""
        docs = list(extractor.extract(resolver))

        # Check specific message texts (order by timestamp)
        texts = [doc.text for doc in docs]
        assert "Hey, did you call the plumber?" in texts
        assert "Yeah, he said he can come Thursday" in texts
        assert "The quote was $350 for the whole job" in texts
        assert "That sounds reasonable, let's go with it" in texts
        assert "Don't forget dinner tomorrow!" in texts
        assert "Happy birthday!! 🎂" in texts

    def test_timestamp_conversion(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that Cocoa epoch nanoseconds are correctly converted."""
        docs = list(extractor.extract(resolver))

        # All timestamps should be datetime objects in February 2026
        for doc in docs:
            assert isinstance(doc.timestamp, datetime)
            assert doc.timestamp.year == 2026
            assert doc.timestamp.month == 2

    def test_timestamp_ordering(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that messages are in chronological order."""
        docs = list(extractor.extract(resolver))

        timestamps = [doc.timestamp for doc in docs]
        assert timestamps == sorted(timestamps)

    def test_is_from_me_flag(self, extractor: MessageExtractor, resolver: ManifestResolver) -> None:
        """Test that is_from_me flag is correctly extracted."""
        docs = list(extractor.extract(resolver))

        is_from_me_values = [doc.metadata["is_from_me"] for doc in docs]
        # Should have mix of True and False
        assert True in is_from_me_values
        assert False in is_from_me_values

    def test_handle_metadata(self, extractor: MessageExtractor, resolver: ManifestResolver) -> None:
        """Test that handle (phone/email) is correctly extracted."""
        docs = list(extractor.extract(resolver))

        handles = set(doc.metadata["handle"] for doc in docs)
        # Fixture has three handles
        assert "+15551234567" in handles
        assert "+15559876543" in handles
        assert "sarah@example.com" in handles

    def test_chat_id_grouping(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that messages are grouped by chat_id."""
        docs = list(extractor.extract(resolver))

        chat_ids = set(doc.metadata["chat_id"] for doc in docs)
        # Fixture has 3 chats
        assert len(chat_ids) == 3
        assert None not in chat_ids  # All messages should have a chat_id

    def test_group_chat_display_name(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that group chat display names are extracted."""
        docs = list(extractor.extract(resolver))

        # Find the message in the "Family Group" chat
        family_group_docs = [doc for doc in docs if doc.metadata.get("chat_name") == "Family Group"]
        assert len(family_group_docs) > 0

    def test_participants_list(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that all chat participants are listed."""
        docs = list(extractor.extract(resolver))

        # Check Sarah conversation (chat_id 3)
        sarah_docs = [
            doc
            for doc in docs
            if doc.metadata["chat_id"] == 3 and doc.metadata["handle"] == "sarah@example.com"
        ]
        if sarah_docs:
            participants = sarah_docs[0].metadata["participants"]
            assert "sarah@example.com" in participants

    def test_source_provenance(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that source provenance is correctly set."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert doc.source.backup_id is not None
            assert doc.source.domain == "HomeDomain"
            assert "Library/SMS/sms.db" in doc.source.relative_path
            assert doc.source.backup_timestamp is not None

    def test_document_id_uniqueness(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that each document gets a unique ID."""
        docs = list(extractor.extract(resolver))

        ids = [doc.id for doc in docs]
        assert len(ids) == len(set(ids))  # All IDs should be unique

    def test_attachment_metadata(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that attachment metadata is correctly handled.

        Note: The test fixture has 0 attachments, but we verify the structure.
        """
        docs = list(extractor.extract(resolver))

        for doc in docs:
            # attachments should be a list (possibly empty)
            assert isinstance(doc.attachments, list)

    def test_emoji_in_message(
        self, extractor: MessageExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that emoji in messages is preserved."""
        docs = list(extractor.extract(resolver))

        # One message should contain the cake emoji
        emoji_messages = [doc for doc in docs if "🎂" in doc.text]
        assert len(emoji_messages) == 1
        assert emoji_messages[0].text == "Happy birthday!! 🎂"

    def test_extractor_implements_protocol(self, extractor: MessageExtractor) -> None:
        """Test that MessageExtractor implements the Extractor protocol."""
        from mudline.models.extractor import Extractor

        assert isinstance(extractor, Extractor)
