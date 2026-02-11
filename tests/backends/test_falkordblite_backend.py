"""Unit tests for FalkorDBLite backend (mocked, no running instance required)."""

import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

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
from tests.backends.conftest import make_memory_node as _make_memory_node

TEST_DB_PATH = "/tmp/test.db"


def _setup_mock(header_names=None, rows=None):
    """Set up mock FalkorDBLite client. Returns (mock_client, mock_graph, mock_FalkorDB_class)."""
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
    """Create a connected FalkorDBLiteBackend. Returns (backend, mock_client, mock_graph, mock_FalkorDB_class)."""
    mock_client, mock_graph, mock_FalkorDB_class = _setup_mock(header_names, rows)
    backend = FalkorDBLiteBackend(db_path=TEST_DB_PATH)
    await backend.connect()
    return backend, mock_client, mock_graph, mock_FalkorDB_class


class TestFalkorDBLiteConnection:

    @pytest.mark.asyncio
    async def test_connect_success(self):
        backend, mock_client, _, mock_FalkorDB = await _connected_backend()

        assert backend._connected is True
        mock_FalkorDB.assert_called_once_with(TEST_DB_PATH)
        mock_client.select_graph.assert_called_once_with('memorygraph')

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        _mock_falkordb_client.FalkorDB = Mock(
            side_effect=Exception("Database file not accessible")
        )
        backend = FalkorDBLiteBackend(db_path='/invalid/path/test.db')

        with pytest.raises(DatabaseConnectionError, match="Database file not accessible"):
            await backend.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        backend, *_ = await _connected_backend()
        await backend.disconnect()
        assert backend._connected is False

    @pytest.mark.asyncio
    async def test_default_path(self):
        _setup_mock()

        backend = FalkorDBLiteBackend()
        await backend.connect()

        call_args = _mock_falkordb_client.FalkorDB.call_args[0]
        assert '.memorygraph/falkordblite.db' in call_args[0]

    def test_backend_name(self):
        assert FalkorDBLiteBackend(db_path=TEST_DB_PATH).backend_name() == "falkordblite"

    def test_supports_fulltext_search(self):
        assert FalkorDBLiteBackend(db_path=TEST_DB_PATH).supports_fulltext_search() is True

    def test_supports_transactions(self):
        assert FalkorDBLiteBackend(db_path=TEST_DB_PATH).supports_transactions() is True


class TestFalkorDBLiteQuery:

    @pytest.mark.asyncio
    async def test_execute_query_read_with_node(self):
        node = _make_node({"id": "123", "title": "Test"})
        backend, *_ = await _connected_backend(header_names=["n"], rows=[[node]])

        result = await backend.execute_query(
            "MATCH (n:Memory {id: $id}) RETURN n",
            parameters={"id": "123"},
            write=False
        )

        assert len(result) == 1
        assert result[0]["n"]["id"] == "123"

    @pytest.mark.asyncio
    async def test_execute_query_write_with_scalar(self):
        backend, *_ = await _connected_backend(header_names=["id"], rows=[["456"]])

        result = await backend.execute_query(
            "CREATE (n:Memory {id: $id}) RETURN n.id as id",
            parameters={"id": "456"},
            write=True
        )

        assert len(result) == 1
        assert result[0]["id"] == "456"

    @pytest.mark.asyncio
    async def test_execute_query_not_connected(self):
        backend = FalkorDBLiteBackend(db_path=TEST_DB_PATH)

        with pytest.raises(DatabaseConnectionError, match="(?i)not connected"):
            await backend.execute_query("MATCH (n) RETURN n")


class TestFalkorDBLiteSchema:

    @pytest.mark.asyncio
    async def test_initialize_schema(self):
        backend, *_ = await _connected_backend()

        backend.execute_query = AsyncMock()
        await backend.initialize_schema()

        assert backend.execute_query.call_count >= 2


class TestFalkorDBLiteMemoryOperations:

    @pytest.fixture
    def sample_memory(self):
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
        backend, *_ = await _connected_backend(
            header_names=["id"], rows=[[sample_memory.id]]
        )
        assert await backend.store_memory(sample_memory) == sample_memory.id

    @pytest.mark.asyncio
    async def test_get_memory(self, sample_memory):
        node = _make_memory_node(
            sample_memory.id,
            title="Redis Timeout Fix",
            content="Increased connection timeout to 5000ms",
            tags=["redis", "timeout", "performance"],
        )
        backend, *_ = await _connected_backend(header_names=["m"], rows=[[node]])

        memory = await backend.get_memory(sample_memory.id)

        assert memory is not None
        assert memory.id == sample_memory.id
        assert memory.title == "Redis Timeout Fix"

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self):
        backend, *_ = await _connected_backend()
        assert await backend.get_memory("nonexistent") is None

    @pytest.mark.asyncio
    async def test_update_memory(self, sample_memory):
        backend, *_ = await _connected_backend(
            header_names=["id"], rows=[[sample_memory.id]]
        )

        sample_memory.title = "Updated Title"
        assert await backend.update_memory(sample_memory) is True

    @pytest.mark.asyncio
    async def test_delete_memory(self, sample_memory):
        backend, _, mock_graph, _ = await _connected_backend()

        exists_result = _make_result(["id"], [[sample_memory.id]])
        delete_result = _make_result([], [])
        mock_graph.query.side_effect = [exists_result, delete_result]

        assert await backend.delete_memory(sample_memory.id) is True


class TestFalkorDBLiteRelationships:

    @pytest.mark.asyncio
    async def test_create_relationship(self):
        rel_id = str(uuid.uuid4())
        backend, *_ = await _connected_backend(header_names=["id"], rows=[[rel_id]])

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
        related_node = _make_memory_node(
            "mem2", title="Related Memory", importance=0.7, confidence=0.8
        )
        rel_props = {"strength": 0.9, "confidence": 0.8, "context": "Test context"}
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

    @pytest.mark.asyncio
    async def test_search_memories(self):
        node = _make_memory_node(
            "search1", title="Redis Timeout", content="Fix for timeout",
            tags=["redis"]
        )
        backend, *_ = await _connected_backend(header_names=["m"], rows=[[node]])

        results = await backend.search_memories(SearchQuery(query="timeout"))

        assert len(results) == 1
        assert results[0].id == "search1"
        assert results[0].title == "Redis Timeout"


class TestFalkorDBLiteStatistics:

    @pytest.mark.asyncio
    async def test_get_memory_statistics(self):
        # Dispatch based on query content; check "m.type" before "COUNT(m)"
        # because the by-type query contains both.
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

    @pytest.mark.asyncio
    async def test_health_check_connected(self):
        backend, *_ = await _connected_backend(header_names=["count"], rows=[[10]])

        health = await backend.health_check()

        assert health["connected"] is True
        assert health["backend_type"] == "falkordblite"
        assert health["statistics"]["memory_count"] == 10
        assert "db_path" in health

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self):
        backend = FalkorDBLiteBackend(db_path=TEST_DB_PATH)

        health = await backend.health_check()

        assert health["connected"] is False
        assert health["backend_type"] == "falkordblite"
