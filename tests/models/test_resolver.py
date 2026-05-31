"""Tests for the ResourceResolver protocol and engine/extractor conformance.

These guard the engine seam: the iOS ``ManifestResolver`` must satisfy the
domain-agnostic ``ResourceResolver`` protocol, and every built-in extractor must
satisfy the ``Extractor`` protocol that now depends only on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mudline.extractors.registry import default_registry
from mudline.foundation.manifest import ManifestResolver
from mudline.models import Extractor, ResourceResolver

if TYPE_CHECKING:
    from pathlib import Path


def test_manifest_resolver_satisfies_resource_resolver(backup_path: Path) -> None:
    """The concrete iOS resolver structurally conforms to the engine protocol."""
    resolver = ManifestResolver(backup_path)
    assert isinstance(resolver, ResourceResolver)


def test_resource_resolver_requires_full_surface() -> None:
    """An object missing a required member must not pass the protocol check."""

    class Partial:
        def resolve(self, domain: str, relative_path: str): ...

    assert not isinstance(Partial(), ResourceResolver)


def test_all_builtin_extractors_satisfy_extractor_protocol() -> None:
    """Every registered extractor conforms to the Extractor protocol."""
    extractors = default_registry.create_all()
    assert extractors, "expected built-in extractors to be registered"
    for extractor in extractors:
        assert isinstance(extractor, Extractor), (
            f"{type(extractor).__name__} does not satisfy Extractor"
        )
