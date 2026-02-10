"""
Unit tests for FalkorDBLite backend implementation.

These tests use mocked FalkorDBLite client to verify backend logic without
requiring a running FalkorDBLite instance.

Mock results use the real FalkorDB response format:
  - result.header: list of [ColumnType, column_name] pairs
  - result.result_set: list of lists (rows of column values)
  - Node objects have a .properties dict
"""

import sys
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

# Create proper mock structure for redislite.falkordb_client
_mock_redislite = MagicMock()
_mock_falkordb_client = MagicMock()
_mock_redislite.falkordb_client = _mock_falkordb_client
sys.modules['redislite'] = _mock_redislite
sys.modules['redislite.falkordb_client'] = _mock_falkordb_client

from memorygraph.backends.falkordblite_backend import FalkorDBLiteBackend
from memorygraph.models import (
    DatabaseConnectionError,
    Memory,
    MemoryType,
    RelationshipProperties,
    RelationshipType,
    SearchQuery,
)
from tests.backends.conftest import make_falkordb_node as _make_node
from tests.backends.conftest import make_falkordb_result as _make_result

TEST_DB_PATH = "/tmp/test.db"


def _make_memory_node(id: str, *, type: str = "solution", title: str = "Test",
                      content: str = "Content", tags: list | None = None,
                      importance: float = 0.8, confidence: float = 0.9,
                      summary: str | None = None) -> Mock:
    """Create a mock FalkorDB node with standard memory properties."""
    now = datetime.now(timezone.utc).isoformat()
    return _make_node({
        "id": id,
        "type": type,
        "title": title,
        "content": content,
        "summary": summary,
        "tags": tags or [],
        "importance": importance,
        "confidence": confidence,
        "created_at": now,
        "updated_at": now,
        "usage_count": 0,
    })


def _setup_mock(header_names=None, rows=None):
    """
    Set up mock FalkorDBLite client with proper result handling.

    Returns:
        Tuple of (mock_client, mock_graph, mock_FalkorDB_class)
    """
    mock_client = Mock()
    mock_graph = Mock()

    if header_names is not None and rows is not None:
        mock_result = _make_result(header_names, rows)
    else:
        mock_result = _make_result([], [])

    mock_graph.query.return_value = mock_result
    mock_client.select_graph.return_value = mock_graph

    mock_FalkorDB_class = Mock(return_value=mock_client)
    _mock_falkordb_client.FalkorDB = mock_FalkorDB_class

    return mock_client, mock_graph, mock_FalkorDB_class


async def _connected_backend(header_names=None, rows=None):
    """Create a connected FalkorDBLiteBackend with mocked internals.

    Returns:
        Tuple of (backend, mock_client, mock_graph, mock_FalkorDB_class)
    """
    mock_client, mock_graph, mock_FalkorDB_class = _setup_mock(header_names, rows)
    backend = FalkorDBLiteBackend(db_path=TEST_DB_PATH)
    await backend.connect()
    return backend, mock_client, mock_graph, mock_FalkorDB_class


class TestFalkorDBLiteConnection:
    """Test FalkorDBLite connection management."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection to FalkorDBLite."""
        backend, mock_client, _, mock_FalkorDB = await _connected_backend()

        assert backend._connected is True
        mock_FalkorDB.assert_called_once_with(TEST_DB_PATH)
        mock_client.select_graph.assert_called_once_with('memorygraph')

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure handling."""
        _mock_falkordb_client.FalkorDB = Mock(
            side_effect=Exception("Database file not accessible")
        )
        backend = FalkorDBLiteBackend(db_path='/invalid/path/test.db')

        with pytest.raises(DatabaseConnectionError, match="Database file not accessible"):
            await backend.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection from FalkorDBLite."""
        backend, *_ = await _connected_backend()
        await backend.disconnect()

        assert backend._connected is False

    @pytest.mark.asyncio
    async def test_default_path(self):
        """Test default database path is used when none specified."""
        _setup_mock()

        backend = FalkorDBLiteBackend()
        await backend.connect()

        call_args = _mock_falkordb_client.FalkorDB.call_args[0]
        assert '.memorygraph/falkordblite.db' in call_args[0]

    def test_backend_name(self):
        """Test backend name identifier."""
        backend = FalkorDBLiteBackend(db_path=TEST_DB_PATH)
        assert backend.backend_name() == "falkordblite"

    def test_supports_fulltext_search(self):
        """Test fulltext search capability reporting."""
        backend = FalkorDBLiteBackend(db_path=TEST_DB_PATH)
        assert backend.supports_fulltext_search() is True

    def test_supports_transactions(self):
        """Test transaction support reporting."""
        backend = FalkorDBLiteBackend(db_path=TEST_DB_PATH)
        assert backend.supports_transactions() is True


class TestFalkorDBLiteQuery:
    """Test FalkorDBLite query execution."""

    @pytest.mark.asyncio
    async def test_execute_query_read_with_node(self):
        """Test executing a read query that returns a Node."""
        node = _make_node({"id": "123", "title": "Test"})
        backend, *_ = await _connected_backend(
            header_names=["n"], rows=[[node]]
        )

        result = await backend.execute_query(
            "MATCH (n:Memory {id: $id}) RETURN n",
            parameters={"id": "123"},
            write=False
        )

        assert len(result) == 1
        assert result[0]["n"]["id"] == "123"

    @pytest.mark.asyncio
    async def test_execute_query_write_with_scalar(self):
        """Test executing a write query that returns scalar values."""
        backend, *_ = await _connected_backend(
            header_names=["id"], rows=[["456"]]
        )

        result = await backend.execute_query(
            "CREATE (n:Memory {id: $id}) RETURN n.id as id",
            parameters={"id": "456"},
            write=True
        )

        assert len(result) == 1
        assert result[0]["id"] == "456"

    @pytest.mark.asyncio
    async def test_execute_query_not_connected(self):
        """Test query execution when not connected."""
        backend = FalkorDBLiteBackend(db_path=TEST_DB_PATH)

        with pytest.raises(DatabaseConnectionError, match="(?i)not connected"):
            await backend.execute_query("MATCH (n) RETURN n")


class TestFalkorDBLiteSchema:
    """Test schema initialization."""

    @pytest.mark.asyncio
    async def test_initialize_schema(self):
        """Test schema creation with constraints and indexes."""
        backend, *_ = await _connected_backend()

        backend.execute_query = AsyncMock()
        await backend.initialize_schema()

        assert backend.execute_query.call_count >= 2


class TestFalkorDBLiteMemoryOperations:
    """Test memory CRUD operations."""

    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory for testing."""
        return Memory(
            id=str(uuid.uuid4()),
            type=MemoryType.SOLUTION,
            title="Redis Timeout Fix",
            content="Increased connection timeout to 5000ms",
            tags=["redis", "timeout", "performance"],
            importance=0.8,
            confidence=0.9
        )

    @pytest.mark.asyncio
    async def test_store_memory(self, sample_memory):
        """Test storing a memory."""
        backend, *_ = await _connected_backend(
            header_names=["id"], rows=[[sample_memory.id]]
        )

        memory_id = await backend.store_memory(sample_memory)

        assert memory_id == sample_memory.id

    @pytest.mark.asyncio
    async def test_get_memory(self, sample_memory):
        """Test retrieving a memory by ID."""
        node = _make_memory_node(
            sample_memory.id,
            title="Redis Timeout Fix",
            content="Increased connection timeout to 5000ms",
            tags=["redis", "timeout", "performance"],
        )
        backend, *_ = await _connected_backend(
            header_names=["m"], rows=[[node]]
        )

        memory = await backend.get_memory(sample_memory.id)

        assert memory is not None
        assert memory.id == sample_memory.id
        assert memory.title == "Redis Timeout Fix"

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self):
        """Test retrieving a non-existent memory."""
        backend, *_ = await _connected_backend()

        memory = await backend.get_memory("nonexistent")

        assert memory is None

    @pytest.mark.asyncio
    async def test_update_memory(self, sample_memory):
        """Test updating an existing memory."""
        backend, *_ = await _connected_backend(
            header_names=["id"], rows=[[sample_memory.id]]
        )

        sample_memory.title = "Updated Title"
        result = await backend.update_memory(sample_memory)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_memory(self, sample_memory):
        """Test deleting a memory."""
        backend, _, mock_graph, _ = await _connected_backend()

        # First call: exists check returns the memory id
        exists_result = _make_result(["id"], [[sample_memory.id]])
        # Second call: DETACH DELETE returns empty
        delete_result = _make_result([], [])
        mock_graph.query.side_effect = [exists_result, delete_result]

        result = await backend.delete_memory(sample_memory.id)

        assert result is True


class TestFalkorDBLiteRelationships:
    """Test relationship operations."""

    @pytest.mark.asyncio
    async def test_create_relationship(self):
        """Test creating a relationship between memories."""
        rel_id = str(uuid.uuid4())
        backend, *_ = await _connected_backend(
            header_names=["id"], rows=[[rel_id]]
        )

        props = RelationshipProperties(strength=0.9, confidence=0.8)
        relationship_id = await backend.create_relationship(
            from_memory_id="mem1",
            to_memory_id="mem2",
            relationship_type=RelationshipType.SOLVES,
            properties=props
        )

        assert relationship_id == rel_id

    @pytest.mark.asyncio
    async def test_get_related_memories(self):
        """Test getting related memories."""
        related_node = _make_memory_node(
            "mem2", title="Related Memory", importance=0.7, confidence=0.8
        )
        rel_props = {
            "strength": 0.9,
            "confidence": 0.8,
            "context": "Test context"
        }
        backend, *_ = await _connected_backend(
            header_names=["related", "rel_type", "rel_props"],
            rows=[[related_node, "SOLVES", rel_props]]
        )

        related = await backend.get_related_memories("mem1")

        assert len(related) == 1
        memory, relationship = related[0]
        assert memory.id == "mem2"
        assert relationship.type == RelationshipType.SOLVES


class TestFalkorDBLiteSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_memories(self):
        """Test searching for memories."""
        node = _make_memory_node(
            "search1", title="Redis Timeout", content="Fix for timeout",
            tags=["redis"]
        )
        backend, *_ = await _connected_backend(
            header_names=["m"], rows=[[node]]
        )

        results = await backend.search_memories(SearchQuery(query="timeout"))

        assert len(results) == 1
        assert results[0].id == "search1"
        assert results[0].title == "Redis Timeout"


class TestFalkorDBLiteStatistics:
    """Test statistics operations."""

    @pytest.mark.asyncio
    async def test_get_memory_statistics(self):
        """Test getting database statistics."""
        # Statistics calls execute_query multiple times with different Cypher.
        # The mock_graph.query dispatches based on query content.
        # Note: memories_by_type query contains both "m.type" AND "COUNT(m)",
        # so check for "m.type" first to avoid false match on COUNT(m)
        def mock_query_side_effect(query, params=None):
            if "m.type" in query:
                return _make_result(
                    ["type", "count"],
                    [["solution", 20], ["problem", 15]]
                )
            elif "COUNT(m)" in query:
                return _make_result(["count"], [[42]])
            elif "COUNT(r)" in query:
                return _make_result(["count"], [[30]])
            elif "AVG(m.importance)" in query:
                return _make_result(["avg_importance"], [[0.75]])
            elif "AVG(m.confidence)" in query:
                return _make_result(["avg_confidence"], [[0.85]])
            else:
                return _make_result([], [])

        backend, _, mock_graph, _ = await _connected_backend()
        mock_graph.query.side_effect = mock_query_side_effect

        stats = await backend.get_memory_statistics()

        assert "total_memories" in stats
        assert "memories_by_type" in stats
        assert stats["memories_by_type"]["solution"] == 20


class TestFalkorDBLiteHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_connected(self):
        """Test health check when connected."""
        backend, *_ = await _connected_backend(
            header_names=["count"], rows=[[10]]
        )

        health = await backend.health_check()

        assert health["connected"] is True
        assert health["backend_type"] == "falkordblite"
        assert health["statistics"]["memory_count"] == 10
        assert "db_path" in health

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self):
        """Test health check when not connected."""
        backend = FalkorDBLiteBackend(db_path=TEST_DB_PATH)

        health = await backend.health_check()

        assert health["connected"] is False
        assert health["backend_type"] == "falkordblite"
