"""Media index for CLIP-based visual semantic search.

This module provides indexing and searching of media (photos/videos) using
CLIP embeddings. Uses lazy imports to allow module loading without open-clip-torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mudline.exceptions import SearchError

logger = logging.getLogger(__name__)


@dataclass
class MediaIndexConfig:
    """Configuration for MediaIndex initialization.

    Args:
        persist_directory: Path where the media index will be stored.
        clip_model: Name of the CLIP model to use (e.g., "ViT-B-32").
        clip_pretrained: Pretrained weights for the CLIP model (e.g., "laion2b_s34b_b79k").
    """

    persist_directory: str = "./data/media_index"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"


class MediaIndex:
    """Index for visual semantic search using CLIP embeddings.

    Stores embeddings of photos and videos, enabling semantic search with
    natural language queries like "sunset beach photos".

    Args:
        config: MediaIndexConfig with persistence and model settings.

    Raises:
        SearchError: If initialization fails or open-clip-torch is unavailable.
    """

    def __init__(self, config: MediaIndexConfig | None = None) -> None:
        """Initialize the media index with the given configuration.

        Args:
            config: MediaIndexConfig instance. Uses defaults if not provided.

        Raises:
            SearchError: If initialization fails.
        """
        self.config = config or MediaIndexConfig()
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_device = None
        self._index: dict[str, dict[str, Any]] = {}  # Simple in-memory index

        # Ensure persist directory exists
        persist_path = Path(self.config.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        try:
            self._initialize_clip()
        except ImportError as e:
            raise SearchError(f"open-clip-torch not installed: {e}") from e
        except Exception as e:
            raise SearchError(f"Failed to initialize CLIP: {e}") from e

    def _initialize_clip(self) -> None:
        """Initialize CLIP model and preprocessing.

        Lazy import of open_clip to avoid hard dependency.

        Raises:
            ImportError: If open-clip-torch is not installed.
            SearchError: If model loading fails.
        """
        try:
            import open_clip
            import torch

            # Load model and preprocessing
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                self.config.clip_model,
                pretrained=self.config.clip_pretrained,
            )

            # Determine device
            self._clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model = self._clip_model.to(self._clip_device)
            self._clip_model.eval()

            logger.info(
                f"Initialized CLIP model '{self.config.clip_model}' on device '{self._clip_device}'"
            )

        except ImportError as e:
            raise ImportError(f"open-clip-torch is required for MediaIndex: {e}") from e
        except Exception as e:
            raise SearchError(f"Failed to load CLIP model: {e}") from e

    def index_photo(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        """Index a single photo by path.

        Embeds the photo using CLIP and stores the embedding with metadata.

        Args:
            path: Path to the photo file.
            metadata: Optional metadata dict to store with the embedding.

        Raises:
            SearchError: If embedding or indexing fails.
        """
        if not path.exists():
            raise SearchError(f"Photo file not found: {path}")

        try:
            import torch
            from PIL import Image

            # Load and preprocess image
            image = Image.open(path).convert("RGB")
            image_tensor = self._clip_preprocess(image).unsqueeze(0).to(self._clip_device)

            # Compute embedding
            with torch.no_grad():
                embedding = self._clip_model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            # Store in index
            photo_id = str(path.resolve())
            self._index[photo_id] = {
                "path": str(path),
                "embedding": embedding.cpu().numpy(),
                "metadata": metadata or {},
            }

            logger.debug(f"Indexed photo: {path}")

        except Exception as e:
            raise SearchError(f"Failed to index photo {path}: {e}") from e

    def search(self, query: str, n: int = 10) -> list[dict[str, Any]]:
        """Search for photos matching a text query.

        Encodes the query using CLIP text encoder and finds the most similar
        photos using cosine similarity.

        Args:
            query: Natural language query (e.g., "sunset beach").
            n: Maximum number of results to return.

        Returns:
            List of dicts with keys: path, score, metadata.

        Raises:
            SearchError: If search fails.
        """
        if not query:
            raise SearchError("Query cannot be empty")

        try:
            import numpy as np
            import torch

            # Encode query
            with torch.no_grad():
                query_embedding = self._clip_model.encode_text(
                    self._tokenizer([query]).to(self._clip_device)
                )
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

            query_emb_np = query_embedding.cpu().numpy()

            # Compute similarities
            results = []
            for _photo_id, photo_data in self._index.items():
                photo_emb = photo_data["embedding"]
                # Cosine similarity
                similarity = float(np.dot(query_emb_np, photo_emb.T).flatten()[0])
                results.append(
                    {
                        "path": photo_data["path"],
                        "score": similarity,
                        "metadata": photo_data["metadata"],
                    }
                )

            # Sort by score descending and return top n
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:n]

        except Exception as e:
            raise SearchError(f"Search failed: {e}") from e

    def count(self) -> int:
        """Return the number of indexed photos.

        Returns:
            Number of photos in the index.
        """
        return len(self._index)
