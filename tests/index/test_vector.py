"""Unit tests for VectorStore class.

Tests use mocking to avoid requiring chromadb installation. The tests verify
that VectorStore correctly prepares data, calls chromadb methods, and handles
errors appropriately.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock

import pytest

from mudline.exceptions import SearchError
from mudline.index.vector import VectorStore, VectorStoreConfig
from mudline.models.document import Document, DocumentType, Source


@pytest.fixture
def mock_chromadb():
    """Fixture to mock chromadb in sys.modules for lazy imports."""
    mock = MagicMock()
    original = sys.modules.get("chromadb")
    sys.modules["chromadb"] = mock
    yield mock
    if original is None:
        sys.modules.pop("chromadb", None)
    else:
        sys.modules["chromadb"] = original


@pytest.fixture
def temp_persist_dir(tmp_path):  # noqa: F821
    """Temporary directory for vector store persistence."""
    return tmp_path / "vector_store"


@pytest.fixture
def vector_config(temp_persist_dir) -> VectorStoreConfig:  # noqa: F821
    """VectorStoreConfig for testing."""
    return VectorStoreConfig(
        persist_directory=temp_persist_dir,
        collection_name="test_documents",
        embedding_model_name="all-MiniLM-L6-v2",
        distance_metric="cosine",
    )


@pytest.fixture
def sample_source() -> Source:
    """Sample Source for test documents."""
    return Source(
        backup_id="test_backup_001",
        domain="HomeDomain",
        relative_path="path/to/file",
        backup_timestamp=datetime.now(),
    )


@pytest.fixture
def sample_documents(sample_source: Source) -> list[Document]:
    """Sample documents for testing."""
    now = datetime.now()
    return [
        Document(
            type=DocumentType.MESSAGE,
            text="Hello, how are you doing?",
            source=sample_source,
            timestamp=now - timedelta(hours=1),
            metadata={"handle": "+15551234567", "chat_id": 1},
        ),
        Document(
            type=DocumentType.NOTE,
            text="Meeting notes: Q1 planning session",
            source=sample_source,
            timestamp=now,
            metadata={"folder": "Work"},
        ),
        Document(
            type=DocumentType.PHOTO,
            text="Vacation photo from Hawaii",
            source=sample_source,
            timestamp=now - timedelta(days=7),
            metadata={"latitude": 20.7965, "longitude": -156.4730},
        ),
    ]


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""

    def test_init_success(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test successful VectorStore initialization."""
        # Mock chromadb client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore(vector_config)

        assert store.config == vector_config
        assert store._client is not None
        assert store._collection is not None
        mock_chromadb.PersistentClient.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()

    def test_init_import_error(self, vector_config: VectorStoreConfig) -> None:
        """Test that SearchError is raised when chromadb import fails."""
        # Replace chromadb with a module that raises ImportError on attribute access
        mock_bad = MagicMock()
        mock_bad.PersistentClient.side_effect = ImportError("No module named 'chromadb'")
        original = sys.modules.get("chromadb")
        sys.modules["chromadb"] = mock_bad
        try:
            with pytest.raises(SearchError, match="ChromaDB"):
                VectorStore(vector_config)
        finally:
            if original:
                sys.modules["chromadb"] = original
            else:
                sys.modules.pop("chromadb", None)

    def test_init_chromadb_error(
        self, mock_chromadb: Mock, vector_config: VectorStoreConfig
    ) -> None:
        """Test that SearchError is raised on chromadb initialization failure."""
        mock_chromadb.PersistentClient.side_effect = Exception("Database locked")
        with pytest.raises(SearchError, match="Failed to initialize ChromaDB"):
            VectorStore(vector_config)

    def test_collection_creation_failure(
        self, mock_chromadb: Mock, vector_config: VectorStoreConfig
    ) -> None:
        """Test that SearchError is raised if collection creation fails."""
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.side_effect = Exception("Schema error")

        with pytest.raises(SearchError, match="Failed to create collection"):
            VectorStore(vector_config)

    def test_persist_directory_created(
        self, mock_chromadb: Mock, temp_persist_dir, vector_config: VectorStoreConfig
    ) -> None:
        """Test that persist directory is created if it doesn't exist."""
        mock_chromadb.PersistentClient.return_value = MagicMock()

        mock_client = mock_chromadb.PersistentClient.return_value
        mock_client.get_or_create_collection.return_value = MagicMock()

        assert not temp_persist_dir.exists()
        VectorStore(vector_config)
        assert temp_persist_dir.exists()


class TestVectorStoreAdd:
    """Tests for the add() method."""

    def test_add_documents(
        self,
        mock_chromadb: Mock,
        vector_config: VectorStoreConfig,
        sample_documents: list[Document],
    ) -> None:
        """Test adding documents to the vector store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore(vector_config)
        store.add(sample_documents)

        # Verify add was called with correct structure
        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        assert call_args is not None

        ids = call_args.kwargs.get("ids") or call_args[0][0]
        documents = call_args.kwargs.get("documents") or call_args[0][1]
        metadatas = call_args.kwargs.get("metadatas") or call_args[0][2]

        assert len(ids) == 3
        assert len(documents) == 3
        assert len(metadatas) == 3

        # Verify document text is preserved
        assert documents[0] == "Hello, how are you doing?"
        assert documents[1] == "Meeting notes: Q1 planning session"

        # Verify metadata contains required fields
        for metadata in metadatas:
            assert "document_type" in metadata
            assert "backup_id" in metadata
            assert "domain" in metadata
            assert "relative_path" in metadata
            assert "has_attachments" in metadata

        # Verify specific metadata fields
        assert metadatas[0]["document_type"] == "message"
        assert metadatas[0]["handle"] == "+15551234567"
        assert metadatas[0]["thread_id"] == "1"
        assert metadatas[1]["document_type"] == "note"

    def test_add_empty_list(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test that adding empty list is a no-op."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore(vector_config)
        store.add([])

        # add should not be called for empty list
        mock_collection.upsert.assert_not_called()

    def test_add_without_timestamp(
        self, mock_chromadb: Mock, vector_config: VectorStoreConfig, sample_source: Source
    ) -> None:
        """Test adding documents without timestamp."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection

        doc = Document(
            type=DocumentType.CONTACT,
            text="John Doe",
            source=sample_source,
            timestamp=None,
        )

        store = VectorStore(vector_config)
        store.add([doc])

        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        metadatas = call_args.kwargs.get("metadatas") or call_args[0][2]
        # timestamp should not be in metadata if not present
        assert "timestamp" not in metadatas[0] or not metadatas[0].get("timestamp")

    def test_add_error_handling(
        self,
        mock_chromadb: Mock,
        vector_config: VectorStoreConfig,
        sample_documents: list[Document],
    ) -> None:
        """Test that add errors are wrapped in SearchError."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.upsert.side_effect = Exception("Database connection lost")

        store = VectorStore(vector_config)
        with pytest.raises(SearchError, match="Failed to add documents"):
            store.add(sample_documents)


class TestVectorStoreQuery:
    """Tests for the query() method."""

    def test_query_success(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test successful query with results."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection

        # Mock query results
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Hello world", "Goodbye world"]],
            "distances": [[0.2, 0.5]],
            "metadatas": [
                [
                    {"document_type": "message", "handle": "+15551234567"},
                    {"document_type": "note"},
                ]
            ],
        }

        store = VectorStore(vector_config)
        results = store.query("Hello", n=10)

        assert len(results) == 2
        doc1, score1 = results[0]
        doc2, score2 = results[1]

        # Check scores are normalized (cosine: 1 - distance/2)
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
        # Lower distance = higher similarity
        assert score1 > score2

        # Check document reconstruction
        assert doc1.type == DocumentType.MESSAGE
        assert doc1.text == "Hello world"
        assert doc1.metadata.get("handle") == "+15551234567"

    def test_query_with_filters(
        self, mock_chromadb: Mock, vector_config: VectorStoreConfig
    ) -> None:
        """Test query with metadata filters."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        store = VectorStore(vector_config)
        where_filter = {"document_type": "message"}
        store.query("Hello", where=where_filter, n=5)

        # Verify where filter was passed
        call_args = mock_collection.query.call_args
        assert call_args.kwargs.get("where") == where_filter or (
            len(call_args[0]) > 3 and call_args[0][3] == where_filter
        )

    def test_query_empty_text(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test that empty query text raises SearchError."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore(vector_config)
        with pytest.raises(SearchError, match="Query text cannot be empty"):
            store.query("")

    def test_query_no_results(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test query with no results."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        store = VectorStore(vector_config)
        results = store.query("xyz")

        assert results == []

    def test_query_error_handling(
        self, mock_chromadb: Mock, vector_config: VectorStoreConfig
    ) -> None:
        """Test that query errors are wrapped in SearchError."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.query.side_effect = Exception("Index corrupted")

        store = VectorStore(vector_config)
        with pytest.raises(SearchError, match="Query failed"):
            store.query("Hello")

    def test_query_distance_normalization_cosine(
        self, mock_chromadb: Mock, vector_config: VectorStoreConfig
    ) -> None:
        """Test that cosine distances are correctly normalized to similarity scores."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection

        # Cosine distance 0 = identical (similarity should be 1)
        # Cosine distance 2 = opposite (similarity should be 0)
        mock_collection.query.return_value = {
            "ids": [["perfect", "opposite"]],
            "documents": [["text1", "text2"]],
            "distances": [[0, 2]],
            "metadatas": [[{}, {}]],
        }

        store = VectorStore(vector_config)
        results = store.query("Hello")

        _, score_perfect = results[0]
        _, score_opposite = results[1]

        assert score_perfect == pytest.approx(1.0)
        assert score_opposite == pytest.approx(0.0)


class TestVectorStorePersistence:
    """Tests for persistence methods."""

    def test_persist_is_noop(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test that persist is a no-op with PersistentClient (auto-persists)."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore(vector_config)
        # Should not raise
        store.persist()


class TestVectorStoreDelete:
    """Tests for the delete() method."""

    def test_delete_documents(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test deleting documents by ID."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore(vector_config)
        store.delete(["doc1", "doc2"])

        mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2"])

    def test_delete_empty_list(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test that deleting empty list is a no-op."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore(vector_config)
        store.delete([])

        mock_collection.delete.assert_not_called()

    def test_delete_error_handling(
        self, mock_chromadb: Mock, vector_config: VectorStoreConfig
    ) -> None:
        """Test that delete errors are wrapped in SearchError."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.delete.side_effect = Exception("Delete failed")

        store = VectorStore(vector_config)
        with pytest.raises(SearchError, match="Failed to delete"):
            store.delete(["doc1"])


class TestVectorStoreClear:
    """Tests for the clear() method."""

    def test_clear_documents(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test clearing all documents from the store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {"ids": ["doc1", "doc2", "doc3"]}

        store = VectorStore(vector_config)
        store.clear()

        mock_collection.get.assert_called_once()
        mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2", "doc3"])

    def test_clear_empty_store(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test clearing an already empty store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {"ids": []}

        store = VectorStore(vector_config)
        store.clear()

        mock_collection.delete.assert_not_called()

    def test_clear_error_handling(
        self, mock_chromadb: Mock, vector_config: VectorStoreConfig
    ) -> None:
        """Test that clear errors are wrapped in SearchError."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.get.side_effect = Exception("Clear failed")

        store = VectorStore(vector_config)
        with pytest.raises(SearchError, match="Failed to clear"):
            store.clear()


class TestVectorStoreCount:
    """Tests for the count() method."""

    def test_count_documents(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test counting documents in the store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.return_value = 3

        store = VectorStore(vector_config)
        count = store.count()

        assert count == 3

    def test_count_empty_store(self, mock_chromadb: Mock, vector_config: VectorStoreConfig) -> None:
        """Test counting in an empty store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.return_value = 0

        store = VectorStore(vector_config)
        count = store.count()

        assert count == 0

    def test_count_error_handling(
        self, mock_chromadb: Mock, vector_config: VectorStoreConfig
    ) -> None:
        """Test that count errors are wrapped in SearchError."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.side_effect = Exception("Count failed")

        store = VectorStore(vector_config)
        with pytest.raises(SearchError, match="Failed to count"):
            store.count()
