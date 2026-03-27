"""PhotoExtractor — extract photos and videos from iOS Camera Roll.

Parses CameraRollDomain/Media/PhotoData/Photos.sqlite to extract photo and video
metadata including location (latitude/longitude), dimensions, media type, and
album information.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from mudline.exceptions import ExtractionError
from mudline.models.document import Document, DocumentType, Source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mudline.foundation.manifest import ManifestResolver

logger = logging.getLogger(__name__)

COCOA_EPOCH = datetime(2001, 1, 1)


def _cocoa_seconds_to_datetime(cocoa_seconds: float) -> datetime:
    """Convert Cocoa epoch seconds to datetime.

    Args:
        cocoa_seconds: Seconds since 2001-01-01.

    Returns:
        Converted datetime.
    """
    return COCOA_EPOCH + timedelta(seconds=cocoa_seconds)


class PhotoExtractor:
    """Extract photos and videos from iOS Camera Roll."""

    @property
    def domain(self) -> str:
        """iOS backup domain for photos."""
        return "CameraRollDomain"

    @property
    def data_type(self) -> str:
        """Document type produced by this extractor."""
        return DocumentType.PHOTO.value

    def can_extract(self, resolver: ManifestResolver) -> bool:
        """Check if the Photos database exists in the backup.

        Args:
            resolver: ManifestResolver for the target backup.

        Returns:
            True if Photos.sqlite exists, False otherwise.
        """
        return resolver.file_exists(
            self.domain, "Media/PhotoData/Photos.sqlite"
        )

    def extract(self, resolver: ManifestResolver) -> Iterator[Document]:
        """Extract all photos and videos from the Photos database.

        Args:
            resolver: ManifestResolver for the target backup.

        Yields:
            Document objects for each photo/video.

        Raises:
            FileNotFoundError: If Photos.sqlite is missing.
            ExtractionError: If the database schema is unexpected.
        """
        try:
            photos_path = resolver.resolve(
                self.domain, "Media/PhotoData/Photos.sqlite"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Photos database not found in backup: "
                f"{self.domain}/Media/PhotoData/Photos.sqlite"
            ) from e

        try:
            conn = sqlite3.connect(
                f"file:{photos_path}?mode=ro&immutable=1", uri=True
            )
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise ExtractionError(f"Failed to open Photos database: {photos_path}") from e

        try:
            # Load album mapping (album_id → album_title)
            albums = self._load_albums(conn)

            # Load asset-to-album mappings
            asset_albums = self._load_asset_albums(conn)

            # Fetch all assets
            cursor = conn.execute(
                """
                SELECT
                    Z_PK, ZFILENAME, ZDATECREATED,
                    ZLATITUDE, ZLONGITUDE, ZWIDTH, ZHEIGHT,
                    ZUNIFORMTYPEIDENTIFIER, ZDIRECTORY
                FROM ZASSET
                ORDER BY ZDATECREATED ASC
                """
            )

            backup_id = self._build_backup_id(resolver)
            backup_timestamp = self._get_backup_timestamp(resolver)

            for row in cursor:
                asset_id = row["Z_PK"]
                filename = row["ZFILENAME"] or ""
                timestamp_cocoa = row["ZDATECREATED"]
                latitude = row["ZLATITUDE"]
                longitude = row["ZLONGITUDE"]
                width = row["ZWIDTH"]
                height = row["ZHEIGHT"]
                type_identifier = row["ZUNIFORMTYPEIDENTIFIER"] or ""
                directory = row["ZDIRECTORY"] or ""

                # Convert timestamp
                timestamp = None
                if timestamp_cocoa is not None:
                    try:
                        timestamp = _cocoa_seconds_to_datetime(timestamp_cocoa)
                    except (ValueError, OverflowError):
                        logger.warning(
                            "Invalid timestamp for asset %d: %s",
                            asset_id,
                            timestamp_cocoa,
                        )

                # Determine media type from type identifier
                # Check for common video types
                media_type = "image"
                video_keywords = ("video", "quicktime", "mpeg", "mp4", "mov")
                if any(kw in type_identifier.lower() for kw in video_keywords):
                    media_type = "video"

                # Get album(s) for this asset
                album_ids = asset_albums.get(asset_id, [])
                album_names = [albums.get(aid, f"Album {aid}") for aid in album_ids]
                album = album_names[0] if album_names else None

                # Handle null location values
                lat = latitude if latitude != 0 else None
                lon = longitude if longitude != 0 else None

                # Build text content
                text = filename or f"{media_type.capitalize()} from {directory}"

                # Build metadata
                metadata = {
                    "latitude": lat,
                    "longitude": lon,
                    "width": width,
                    "height": height,
                    "album": album,
                    "media_type": media_type,
                }

                # Create source
                source = Source(
                    backup_id=backup_id,
                    domain=self.domain,
                    relative_path=f"Media/PhotoData/Photos.sqlite/asset/{asset_id}",
                    backup_timestamp=backup_timestamp,
                )

                # Create document
                doc = Document(
                    type=DocumentType.PHOTO,
                    text=text,
                    timestamp=timestamp,
                    metadata=metadata,
                    source=source,
                )

                yield doc

        except sqlite3.Error as e:
            raise ExtractionError(f"Database error while extracting photos: {e}") from e
        finally:
            conn.close()

    def _load_albums(self, conn: sqlite3.Connection) -> dict[int, str]:
        """Load all albums from the database.

        Args:
            conn: SQLite connection to Photos.sqlite.

        Returns:
            Mapping of album Z_PK to album title.
        """
        albums: dict[int, str] = {}
        try:
            cursor = conn.execute(
                "SELECT Z_PK, ZTITLE FROM ZGENERICALBUM WHERE ZTITLE IS NOT NULL"
            )
            for row in cursor:
                albums[row[0]] = row[1]
        except sqlite3.Error as e:
            logger.warning("Failed to load albums: %s", e)
        return albums

    def _load_asset_albums(self, conn: sqlite3.Connection) -> dict[int, list[int]]:
        """Load mapping of assets to their albums.

        Args:
            conn: SQLite connection to Photos.sqlite.

        Returns:
            Mapping of asset Z_PK to list of album Z_PKs.
        """
        asset_albums: dict[int, list[int]] = {}
        try:
            cursor = conn.execute(
                "SELECT Z_26ALBUMS, Z_34ASSETS FROM Z_26ASSETS"
            )
            for row in cursor:
                album_id = row[0]
                asset_id = row[1]
                if asset_id not in asset_albums:
                    asset_albums[asset_id] = []
                asset_albums[asset_id].append(album_id)
        except sqlite3.Error as e:
            logger.warning("Failed to load asset-album mappings: %s", e)
        return asset_albums

    def _build_backup_id(self, resolver: ManifestResolver) -> str:
        """Build a backup ID from backup path.

        Args:
            resolver: ManifestResolver.

        Returns:
            A string identifier for the backup.
        """
        return resolver.backup_path.name

    def _get_backup_timestamp(self, resolver: ManifestResolver) -> datetime:
        """Get the backup timestamp from Info.plist.

        Args:
            resolver: ManifestResolver.

        Returns:
            Backup timestamp, or current time if unavailable.
        """
        import plistlib

        info_plist = resolver.backup_path / "Info.plist"
        if info_plist.exists():
            try:
                with open(info_plist, "rb") as f:
                    info = plistlib.load(f)
                    if "Last Backup Date" in info:
                        return info["Last Backup Date"]
            except Exception as e:
                logger.warning("Failed to read Info.plist: %s", e)

        return datetime.now()
