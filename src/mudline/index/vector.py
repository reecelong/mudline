"""Vector store for semantic search over documents using ChromaDB with persistent storage.

The VectorStore class embeds documents using a configurable sentence-transformer model
and stores embeddings in ChromaDB with persistence. Lazy imports allow the module
to be imported even if chromadb is not installed (useful for testing).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mudline.exceptions import SearchError
from mudline.models.document import Document

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for VectorStore initialization.

    Args:
        persist_directory: Path where ChromaDB will store embeddings and metadata.
        collection_name: Name of the ChromaDB collection to use.
        embedding_model_name: Name of the sentence-transformer model for embeddings.
        distance_metric: Distance metric for similarity search ("cosine", "l2", "ip").
    """

    persist_directory: Path  # noqa: F821
    collection_name: str = "documents"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    distance_metric: str = "cosine"


class VectorStore:
    """Persistent vector store for document embeddings using ChromaDB.

    Stores documents with their embeddings and metadata, enabling semantic search.
    Lazily imports chromadb to allow module loading without the library installed.

    Args:
        config: VectorStoreConfig with persistence and model settings.

    Raises:
        SearchError: If ChromaDB operations fail or the library is unavailable.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        """Initialize the vector store with the given configuration.

        Args:
            config: VectorStoreConfig instance.

        Raises:
            SearchError: If ChromaDB initialization fails.
        """
        self.config = config
        self._client = None
        self._collection = None
        self._embedding_fn = None

        # Ensure persist directory exists
        self.config.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            self._initialize_chromadb()
        except ImportError as e:
            raise SearchError(f"ChromaDB not installed: {e}") from e
        except Exception as e:
            raise SearchError(f"Failed to initialize ChromaDB: {e}") from e

    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and collection.

        Lazy import of chromadb to avoid hard dependency.

        Raises:
            ImportError: If chromadb is not installed.
            SearchError: If collection creation fails.
        """
        import chromadb

        # Create persistent client using the current ChromaDB API
        self._client = chromadb.PersistentClient(
            path=str(self.config.persist_directory),
        )

        # Create or get collection
        # Collection names must be 3-63 chars, alphanumeric + dash/underscore
        sanitized_name = self.config.collection_name.replace(" ", "_").lower()[:63]
        try:
            self._collection = self._client.get_or_create_collection(
                name=sanitized_name,
                metadata={
                    "hnsw:space": self.config.distance_metric,
                },
            )
        except Exception as e:
            raise SearchError(f"Failed to create collection {sanitized_name}: {e}") from e

        logger.info(
            f"Initialized ChromaDB collection '{sanitized_name}' at {self.config.persist_directory}"
        )

    def add(self, docs: list[Document]) -> None:
        """Embed and store documents in the vector store.

        Extracts text from documents, generates embeddings, and stores them
        with metadata for filtering. IDs are deterministic based on document content.

        Args:
            docs: List of Document objects to embed and store.

        Raises:
            SearchError: If embedding or storage fails.
        """
        if not docs:
            return

        try:
            # Prepare chromadb insert format
            ids = []
            documents = []
            metadatas = []

            for doc in docs:
                ids.append(doc.id)
                documents.append(doc.text)

                # Build metadata dict with key fields for filtering
                metadata: dict[str, Any] = {
                    "document_type": doc.type.value,
                    "backup_id": doc.source.backup_id,
                    "domain": doc.source.domain,
                    "relative_path": doc.source.relative_path,
                    "has_attachments": len(doc.attachments) > 0,
                }

                # Add timestamp if present
                if doc.timestamp:
                    metadata["timestamp"] = doc.timestamp.isoformat()

                # Add contact handle if present in metadata
                if "handle" in doc.metadata:
                    metadata["handle"] = doc.metadata["handle"]

                # Add thread_id if present
                if "chat_id" in doc.metadata:
                    metadata["thread_id"] = str(doc.metadata["chat_id"])

                metadatas.append(metadata)

            # Deduplicate within the batch — ChromaDB rejects duplicate IDs
            seen: set[str] = set()
            deduped_ids = []
            deduped_documents = []
            deduped_metadatas = []
            for i, doc_id in enumerate(ids):
                if doc_id not in seen:
                    seen.add(doc_id)
                    deduped_ids.append(doc_id)
                    deduped_documents.append(documents[i])
                    deduped_metadatas.append(metadatas[i])

            if not deduped_ids:
                return

            # Use upsert to handle cross-batch duplicates gracefully
            self._collection.upsert(
                ids=deduped_ids, documents=deduped_documents, metadatas=deduped_metadatas
            )
            logger.debug(f"Upserted {len(deduped_ids)} documents to vector store")

        except Exception as e:
            raise SearchError(f"Failed to add documents to vector store: {e}") from e

    def query(
        self,
        text: str,
        where: dict[str, Any] | None = None,
        n: int = 10,
    ) -> list[tuple[Document, float]]:
        """Query the vector store for documents similar to the given text.

        Performs semantic search using embeddings with optional metadata filtering.

        Args:
            text: Query text to search for.
            where: Optional ChromaDB where filter for metadata filtering.
            n: Maximum number of results to return.

        Returns:
            List of (Document, score) tuples sorted by descending score.
            Documents are reconstructed from stored metadata and text.

        Raises:
            SearchError: If the query fails.
        """
        if not text:
            raise SearchError("Query text cannot be empty")

        try:
            # Query ChromaDB
            where_filter = where if where is not None else {}
            results = self._collection.query(
                query_texts=[text],
                n_results=n,
                where=where_filter if where_filter else None,
            )

            # Reconstruct Document objects from results
            documents_with_scores = []

            if results and results["ids"] and len(results["ids"]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    score = (
                        results["distances"][0][i]
                        if results["distances"] and results["distances"][0]
                        else 0.0
                    )
                    metadata = (
                        results["metadatas"][0][i]
                        if results["metadatas"] and results["metadatas"][0]
                        else {}
                    )
                    text_content = (
                        results["documents"][0][i]
                        if results["documents"] and results["documents"][0]
                        else ""
                    )

                    # Normalize distance to similarity score (0-1, higher=better)
                    # For cosine distance, 0 = identical, 2 = completely different
                    if self.config.distance_metric == "cosine":
                        similarity = 1 - (score / 2)
                    else:
                        # For other metrics, use inverse
                        similarity = 1 / (1 + score)

                    # Reconstruct minimal Document for results
                    # Note: This is a simplified Document; full reconstruction
                    # would require fetching from the SQL store
                    doc = self._reconstruct_document(doc_id, text_content, metadata)
                    documents_with_scores.append((doc, similarity))

            logger.debug(f"Query returned {len(documents_with_scores)} results")
            return documents_with_scores

        except Exception as e:
            raise SearchError(f"Query failed: {e}") from e

    def _reconstruct_document(self, doc_id: str, text: str, metadata: dict[str, Any]) -> Document:
        """Reconstruct a Document object from stored metadata and text.

        Args:
            doc_id: The document ID.
            text: The document text content.
            metadata: The stored metadata dict.

        Returns:
            A reconstructed Document object.
        """
        from datetime import datetime

        from mudline.models.document import DocumentType, Source

        # Extract metadata fields
        doc_type = DocumentType(metadata.get("document_type", "note"))
        backup_id = metadata.get("backup_id", "")
        domain = metadata.get("domain", "")
        relative_path = metadata.get("relative_path", "")

        # Parse timestamp if present
        timestamp = None
        if "timestamp" in metadata and metadata["timestamp"]:
            with contextlib.suppress(ValueError, TypeError):
                timestamp = datetime.fromisoformat(metadata["timestamp"])

        # Reconstruct source
        source = Source(
            backup_id=backup_id,
            domain=domain,
            relative_path=relative_path,
            backup_timestamp=datetime.now(),  # Placeholder; ideally stored
        )

        # Reconstruct metadata dict from stored fields
        doc_metadata = {}
        if "handle" in metadata:
            doc_metadata["handle"] = metadata["handle"]
        if "thread_id" in metadata:
            with contextlib.suppress(ValueError, TypeError):
                doc_metadata["chat_id"] = int(metadata["thread_id"])

        # Reconstruct Document
        doc = Document(
            type=doc_type,
            text=text,
            source=source,
            timestamp=timestamp,
            metadata=doc_metadata,
            attachments=[],  # Attachments not stored in vector metadata
            id=doc_id,
        )
        return doc

    def persist(self) -> None:
        """Explicitly persist the vector store to disk.

        PersistentClient auto-persists on every write, so this is a no-op
        kept for API compatibility.
        """
        logger.debug("Vector store uses PersistentClient — auto-persisted")

    def delete(self, doc_ids: list[str]) -> None:
        """Delete documents from the vector store by ID.

        Args:
            doc_ids: List of document IDs to delete.

        Raises:
            SearchError: If deletion fails.
        """
        if not doc_ids:
            return

        try:
            self._collection.delete(ids=doc_ids)
            logger.debug(f"Deleted {len(doc_ids)} documents from vector store")
        except Exception as e:
            raise SearchError(f"Failed to delete documents: {e}") from e

    def clear(self) -> None:
        """Clear all documents from the collection.

        Raises:
            SearchError: If clearing fails.
        """
        try:
            # ChromaDB doesn't have a native clear, so delete by querying all
            results = self._collection.get()
            if results and results["ids"]:
                self._collection.delete(ids=results["ids"])
                logger.debug("Vector store cleared")
        except Exception as e:
            raise SearchError(f"Failed to clear vector store: {e}") from e

    def count(self) -> int:
        """Return the number of documents in the vector store.

        Returns:
            Number of stored documents.

        Raises:
            SearchError: If the count query fails.
        """
        try:
            return self._collection.count()
        except Exception as e:
            raise SearchError(f"Failed to count documents: {e}") from e
