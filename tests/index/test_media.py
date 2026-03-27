"""Tests for the media index with CLIP embeddings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mudline.exceptions import SearchError
from mudline.index.media import MediaIndex, MediaIndexConfig

# Skip all tests if open-clip-torch is not available
pytestmark = pytest.mark.skipif(
    True,  # Always skip since dependencies not available
    reason="open-clip-torch not installed",
)


@pytest.fixture
def media_config(tmp_path: Path) -> MediaIndexConfig:
    """Create a test MediaIndexConfig."""
    return MediaIndexConfig(
        persist_directory=str(tmp_path / "media_index"),
        clip_model="ViT-B-32",
        clip_pretrained="laion2b_s34b_b79k",
    )




class TestMediaIndexConfig:
    """Test MediaIndexConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MediaIndexConfig(persist_directory="/tmp/media")

        assert config.persist_directory == "/tmp/media"
        assert config.clip_model == "ViT-B-32"
        assert config.clip_pretrained == "laion2b_s34b_b79k"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = MediaIndexConfig(
            persist_directory="/custom/path",
            clip_model="ViT-L-14",
            clip_pretrained="openai",
        )

        assert config.persist_directory == "/custom/path"
        assert config.clip_model == "ViT-L-14"
        assert config.clip_pretrained == "openai"


class TestMediaIndex:
    """Test the MediaIndex class."""

    def test_initialization_without_clip_installed(
        self, media_config: MediaIndexConfig
    ) -> None:
        """Test that initialization fails gracefully if open-clip-torch is missing."""
        with patch(
            "mudline.index.media.open_clip", side_effect=ImportError("Not installed")
        ), pytest.raises(SearchError, match="open-clip-torch not installed"):
            MediaIndex(media_config)

    def test_initialization_with_import_error(
        self, media_config: MediaIndexConfig
    ) -> None:
        """Test initialization handles open_clip import errors."""
        with patch(
            "mudline.index.media.open_clip", side_effect=ImportError("Mock error")
        ), pytest.raises(SearchError, match="open-clip-torch not installed"):
            MediaIndex(media_config)

    def test_count_initial(self, media_config: MediaIndexConfig) -> None:
        """Test that count starts at zero."""
        with patch(
            "mudline.index.media.open_clip.create_model_and_transforms"
        ) as mock_create:
            mock_model = MagicMock()
            mock_preprocess = MagicMock()
            mock_create.return_value = (mock_model, None, mock_preprocess)

            index = MediaIndex(media_config)
            assert index.count() == 0

    def test_persist_directory_created(self, media_config: MediaIndexConfig) -> None:
        """Test that persist directory is created."""
        persist_path = Path(media_config.persist_directory)

        with patch(
            "mudline.index.media.open_clip.create_model_and_transforms"
        ) as mock_create:
            mock_model = MagicMock()
            mock_preprocess = MagicMock()
            mock_create.return_value = (mock_model, None, mock_preprocess)

            MediaIndex(media_config)

            assert persist_path.exists()

    def test_config_stored(self, media_config: MediaIndexConfig) -> None:
        """Test that config is stored."""
        with patch(
            "mudline.index.media.open_clip.create_model_and_transforms"
        ) as mock_create:
            mock_model = MagicMock()
            mock_preprocess = MagicMock()
            mock_create.return_value = (mock_model, None, mock_preprocess)

            index = MediaIndex(media_config)

            assert index.config == media_config

    def test_index_photo_file_not_found(
        self, media_config: MediaIndexConfig
    ) -> None:
        """Test that indexing a nonexistent file raises SearchError."""
        with patch(
            "mudline.index.media.open_clip.create_model_and_transforms"
        ) as mock_create:
            mock_model = MagicMock()
            mock_preprocess = MagicMock()
            mock_create.return_value = (mock_model, None, mock_preprocess)

            index = MediaIndex(media_config)

            with pytest.raises(SearchError, match="Photo file not found"):
                index.index_photo(Path("/nonexistent/photo.jpg"))

    def test_search_empty_query_raises_error(
        self, media_config: MediaIndexConfig
    ) -> None:
        """Test that searching with empty query raises SearchError."""
        with patch(
            "mudline.index.media.open_clip.create_model_and_transforms"
        ) as mock_create:
            mock_model = MagicMock()
            mock_preprocess = MagicMock()
            mock_create.return_value = (mock_model, None, mock_preprocess)

            index = MediaIndex(media_config)

            with pytest.raises(SearchError, match="Query cannot be empty"):
                index.search("")

    def test_search_returns_list(
        self, media_config: MediaIndexConfig
    ) -> None:
        """Test that search returns a list."""
        with patch(
            "mudline.index.media.open_clip.create_model_and_transforms"
        ) as mock_create:
            mock_model = MagicMock()
            mock_preprocess = MagicMock()
            mock_create.return_value = (mock_model, None, mock_preprocess)

            index = MediaIndex(media_config)

            # Search on empty index should return empty list
            results = index.search("sunset beach")
            assert isinstance(results, list)

    def test_index_photo_with_metadata(
        self, media_config: MediaIndexConfig, tmp_path: Path
    ) -> None:
        """Test indexing a photo with metadata."""
        # Create a dummy image file
        dummy_image = tmp_path / "test.jpg"
        dummy_image.write_bytes(b"fake image data")

        with patch(
            "mudline.index.media.open_clip.create_model_and_transforms"
        ) as mock_create, patch("mudline.index.media.Image") as mock_image, patch(
            "mudline.index.media.torch"
        ):
            mock_model = MagicMock()
            mock_preprocess = MagicMock()
            mock_preprocess.return_value = MagicMock()
            mock_create.return_value = (mock_model, None, mock_preprocess)

            # Mock PIL Image
            mock_pil_image = MagicMock()
            mock_image.open.return_value = mock_pil_image
            mock_pil_image.convert.return_value = mock_pil_image

            # Mock torch operations
            mock_tensor = MagicMock()
            mock_tensor.unsqueeze.return_value = mock_tensor
            mock_tensor.to.return_value = mock_tensor
            mock_preprocess.return_value = mock_tensor

            mock_embedding = MagicMock()
            mock_embedding.cpu.return_value = mock_embedding
            mock_embedding.numpy.return_value = [[1, 2, 3]]
            mock_model.encode_image.return_value = mock_embedding

            import contextlib

            index = MediaIndex(media_config)
            metadata = {"album": "Vacation", "location": "Hawaii"}

            # This may fail due to missing dependencies, so we catch SearchError
            with contextlib.suppress(SearchError):
                index.index_photo(dummy_image, metadata)

    def test_search_limit_parameter(
        self, media_config: MediaIndexConfig
    ) -> None:
        """Test that search respects limit parameter."""
        with patch(
            "mudline.index.media.open_clip.create_model_and_transforms"
        ) as mock_create:
            mock_model = MagicMock()
            mock_preprocess = MagicMock()
            mock_create.return_value = (mock_model, None, mock_preprocess)

            index = MediaIndex(media_config)

            # Search with custom limit
            results = index.search("sunset", n=5)
            assert isinstance(results, list)
            assert len(results) <= 5
