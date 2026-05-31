"""Tests for PhotoExtractor."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from mudline.extractors.photos import PhotoExtractor
from mudline.foundation.manifest import ManifestResolver
from mudline.models.document import DocumentType

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def extractor() -> PhotoExtractor:
    """Create a PhotoExtractor instance."""
    return PhotoExtractor()


@pytest.fixture
def temp_backup(tmp_path: Path) -> Path:
    """Create a minimal temporary backup with Photos.sqlite."""
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()

    # Create hash prefix directories
    for i in range(256):
        (backup_dir / f"{i:02x}").mkdir()

    # Create Manifest.db
    manifest_path = backup_dir / "Manifest.db"
    manifest_conn = sqlite3.connect(manifest_path)
    manifest_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Files (
            fileID TEXT PRIMARY KEY,
            domain TEXT,
            relativePath TEXT,
            flags INTEGER
        )
        """
    )

    # Create Photos.sqlite in the backup
    photos_file_id = "aabbccddee1122334455667788990011"
    photos_rel_path = "Media/PhotoData/Photos.sqlite"
    manifest_conn.execute(
        "INSERT INTO Files (fileID, domain, relativePath, flags) VALUES (?, ?, ?, ?)",
        (photos_file_id, "CameraRollDomain", photos_rel_path, 1),
    )
    manifest_conn.commit()
    manifest_conn.close()

    # Create the actual Photos.sqlite file
    photos_dir = backup_dir / photos_file_id[:2]
    photos_dir.mkdir(exist_ok=True)
    photos_path = photos_dir / photos_file_id

    photos_conn = sqlite3.connect(photos_path)
    photos_conn.execute(
        """
        CREATE TABLE ZASSET (
            Z_PK INTEGER PRIMARY KEY,
            ZFILENAME TEXT,
            ZDATECREATED REAL,
            ZLATITUDE REAL,
            ZLONGITUDE REAL,
            ZWIDTH INTEGER,
            ZHEIGHT INTEGER,
            ZUNIFORMTYPEIDENTIFIER TEXT,
            ZDIRECTORY TEXT
        )
        """
    )
    photos_conn.execute(
        """
        CREATE TABLE ZGENERICALBUM (
            Z_PK INTEGER PRIMARY KEY,
            ZTITLE TEXT
        )
        """
    )
    photos_conn.execute(
        """
        CREATE TABLE Z_26ASSETS (
            Z_26ALBUMS INTEGER,
            Z_34ASSETS INTEGER
        )
        """
    )

    # Insert sample photos
    cocoa_epoch_2026_02_15 = 792806400.0  # Feb 15, 2026 in Cocoa epoch
    photos_conn.execute(
        "INSERT INTO ZASSET VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "photo1.jpg",
            cocoa_epoch_2026_02_15,
            37.7749,
            -122.4194,
            1920,
            1080,
            "public.jpeg",
            "DCIM/100APPLE",
        ),
    )
    photos_conn.execute(
        "INSERT INTO ZASSET VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            2,
            "video1.mp4",
            cocoa_epoch_2026_02_15 + 3600,
            None,
            None,
            1280,
            720,
            "public.mpeg-4",
            "DCIM/100APPLE",
        ),
    )

    # Insert albums
    photos_conn.execute(
        "INSERT INTO ZGENERICALBUM VALUES (?, ?)",
        (1, "Favorites"),
    )

    # Link assets to albums
    photos_conn.execute(
        "INSERT INTO Z_26ASSETS VALUES (?, ?)",
        (1, 1),
    )

    photos_conn.commit()
    photos_conn.close()

    # Create Info.plist
    import plistlib

    info_plist = backup_dir / "Info.plist"
    with open(info_plist, "wb") as f:
        plistlib.dump({"Last Backup Date": datetime(2026, 2, 15, 10, 0, 0)}, f)

    return backup_dir


@pytest.fixture
def resolver(temp_backup: Path) -> ManifestResolver:
    """Create a ManifestResolver for the temporary backup."""
    return ManifestResolver(temp_backup)


class TestPhotoExtractor:
    """Test suite for PhotoExtractor."""

    def test_domain_property(self, extractor: PhotoExtractor) -> None:
        """Test that domain property is correct."""
        assert extractor.domain == "CameraRollDomain"

    def test_data_type_property(self, extractor: PhotoExtractor) -> None:
        """Test that data_type property is correct."""
        assert extractor.data_type == "photo"

    def test_can_extract_with_photos_db(
        self, extractor: PhotoExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that can_extract returns True when Photos.sqlite exists."""
        assert extractor.can_extract(resolver) is True

    def test_can_extract_without_photos_db(self, extractor: PhotoExtractor, tmp_path: Path) -> None:
        """Test that can_extract returns False when Photos.sqlite is missing."""
        # Create a minimal backup without Photos.sqlite
        backup_dir = tmp_path / "empty_backup"
        backup_dir.mkdir()
        (backup_dir / "00").mkdir()

        manifest_path = backup_dir / "Manifest.db"
        manifest_conn = sqlite3.connect(manifest_path)
        manifest_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS Files (
                fileID TEXT PRIMARY KEY,
                domain TEXT,
                relativePath TEXT,
                flags INTEGER
            )
            """
        )
        manifest_conn.commit()
        manifest_conn.close()

        resolver = ManifestResolver(backup_dir)
        assert extractor.can_extract(resolver) is False

    def test_extract_photos(self, extractor: PhotoExtractor, resolver: ManifestResolver) -> None:
        """Test extracting photos from the temporary backup."""
        docs = list(extractor.extract(resolver))

        # Should have 2 assets (1 photo + 1 video)
        assert len(docs) == 2

        # All should be PHOTO type
        assert all(doc.type == DocumentType.PHOTO for doc in docs)

    def test_photo_metadata(self, extractor: PhotoExtractor, resolver: ManifestResolver) -> None:
        """Test that photo metadata is correctly extracted."""
        docs = list(extractor.extract(resolver))

        # First doc should be the photo
        photo = docs[0]
        assert photo.metadata["media_type"] == "image"
        assert photo.metadata["width"] == 1920
        assert photo.metadata["height"] == 1080
        assert photo.metadata["latitude"] == pytest.approx(37.7749)
        assert photo.metadata["longitude"] == pytest.approx(-122.4194)
        assert photo.metadata["album"] == "Favorites"

    def test_video_metadata(self, extractor: PhotoExtractor, resolver: ManifestResolver) -> None:
        """Test that video metadata is correctly extracted."""
        docs = list(extractor.extract(resolver))

        # Second doc should be the video
        video = docs[1]
        assert video.metadata["media_type"] == "video"
        assert video.metadata["width"] == 1280
        assert video.metadata["height"] == 720
        assert video.metadata["latitude"] is None
        assert video.metadata["longitude"] is None

    def test_timestamp_conversion(
        self, extractor: PhotoExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that Cocoa epoch seconds are correctly converted."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert isinstance(doc.timestamp, datetime)
            assert doc.timestamp.year == 2026
            assert doc.timestamp.month == 2

    def test_source_provenance(self, extractor: PhotoExtractor, resolver: ManifestResolver) -> None:
        """Test that source provenance is correctly set."""
        docs = list(extractor.extract(resolver))

        for doc in docs:
            assert doc.source.backup_id is not None
            assert doc.source.domain == "CameraRollDomain"
            assert "Media/PhotoData/Photos.sqlite" in doc.source.relative_path
            assert doc.source.backup_timestamp is not None

    def test_document_id_uniqueness(
        self, extractor: PhotoExtractor, resolver: ManifestResolver
    ) -> None:
        """Test that each document gets a unique ID."""
        docs = list(extractor.extract(resolver))

        ids = [doc.id for doc in docs]
        assert len(ids) == len(set(ids))  # All IDs should be unique

    def test_extractor_implements_protocol(self, extractor: PhotoExtractor) -> None:
        """Test that PhotoExtractor implements the Extractor protocol."""
        from mudline.models.extractor import Extractor

        assert isinstance(extractor, Extractor)
