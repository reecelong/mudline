"""Tests for the extractor registry."""

from __future__ import annotations

import pytest

from mudline.extractors.registry import ExtractorRegistry, default_registry


class DummyExtractor:
    """Dummy extractor for testing."""

    @property
    def domain(self) -> str:
        return "TestDomain"

    @property
    def data_type(self) -> str:
        return "test"

    def extract(self, resolver):
        yield from []

    def can_extract(self, resolver) -> bool:
        return True


class TestExtractorRegistry:
    """Test the ExtractorRegistry class."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving an extractor."""
        registry = ExtractorRegistry()
        registry.register("test", DummyExtractor)

        assert registry.get("test") == DummyExtractor

    def test_get_nonexistent_raises_keyerror(self) -> None:
        """Test that getting a non-existent extractor raises KeyError."""
        registry = ExtractorRegistry()

        with pytest.raises(KeyError, match="not found in registry"):
            registry.get("nonexistent")

    def test_list_returns_sorted_names(self) -> None:
        """Test that list returns all registered names in sorted order."""
        registry = ExtractorRegistry()
        registry.register("charlie", DummyExtractor)
        registry.register("alice", DummyExtractor)
        registry.register("bob", DummyExtractor)

        assert registry.list() == ["alice", "bob", "charlie"]

    def test_list_empty_registry(self) -> None:
        """Test that list returns empty list for empty registry."""
        registry = ExtractorRegistry()
        assert registry.list() == []

    def test_create_all(self) -> None:
        """Test creating instances of all registered extractors."""
        registry = ExtractorRegistry()
        registry.register("test1", DummyExtractor)
        registry.register("test2", DummyExtractor)

        extractors = registry.create_all()

        assert len(extractors) == 2
        assert all(isinstance(e, DummyExtractor) for e in extractors)

    def test_create_all_empty_registry(self) -> None:
        """Test create_all on empty registry returns empty list."""
        registry = ExtractorRegistry()
        assert registry.create_all() == []

    def test_default_registry_has_extractors(self) -> None:
        """Test that default_registry is pre-populated."""
        # Should have at least some extractors registered
        registered = default_registry.list()
        assert len(registered) > 0

        # Check for expected extractors
        expected = {
            "messages",
            "contacts",
            "photos",
            "notes",
            "calendar",
            "calls",
            "safari",
            "voicemail",
        }
        registered_set = set(registered)
        # At least some of these should be present
        assert len(expected & registered_set) > 0

    def test_default_registry_can_create_instances(self) -> None:
        """Test that default_registry can create instances of its extractors."""
        extractors = default_registry.create_all()

        # Should have created some extractors
        assert len(extractors) > 0

        # Each should be an Extractor
        for extractor in extractors:
            assert hasattr(extractor, "domain")
            assert hasattr(extractor, "data_type")
            assert hasattr(extractor, "extract")

    def test_register_overwrites_existing(self) -> None:
        """Test that registering with an existing name overwrites it."""
        registry = ExtractorRegistry()

        class ExtractorV1:
            version = 1

        class ExtractorV2:
            version = 2

        registry.register("test", ExtractorV1)
        assert registry.get("test").version == 1  # type: ignore

        registry.register("test", ExtractorV2)
        assert registry.get("test").version == 2  # type: ignore
