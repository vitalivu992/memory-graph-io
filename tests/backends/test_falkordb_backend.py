"""Unit tests for FalkorDB backend (mocked, no running instance required)."""

import sys
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.modules['falkordb'] = MagicMock()

from memorygraph.backends.falkordb_backend import FalkorDBBackend
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


class TestFalkorDBConnection:

    @pytest.mark.asyncio
    async def test_connect_success(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            result = await backend.connect()

            assert result is True
            assert backend._connected is True
            mock_client.select_graph.assert_called_once_with('memorygraph')

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_falkordb_class.side_effect = Exception("Connection refused")

            backend = FalkorDBBackend(host='localhost', port=6379)

            with pytest.raises(DatabaseConnectionError, match="Connection refused"):
                await backend.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()
            await backend.disconnect()

            assert backend._connected is False

    def test_backend_name(self):
        backend = FalkorDBBackend(host='localhost', port=6379)
        assert backend.backend_name() == "falkordb"

    def test_supports_fulltext_search(self):
        backend = FalkorDBBackend(host='localhost', port=6379)
        assert backend.supports_fulltext_search() is True

    def test_supports_transactions(self):
        backend = FalkorDBBackend(host='localhost', port=6379)
        assert backend.supports_transactions() is True


class TestFalkorDBQuery:

    @pytest.mark.asyncio
    async def test_execute_query_read_with_node(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

            node = _make_node({"id": "123", "title": "Test"})
            mock_graph.query.return_value = _make_result(["n"], [[node]])
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            result = await backend.execute_query(
                "MATCH (n:Memory {id: $id}) RETURN n",
                parameters={"id": "123"},
                write=False
            )

            assert len(result) == 1
            assert result[0]["n"]["id"] == "123"

    @pytest.mark.asyncio
    async def test_execute_query_write_with_scalar(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

            mock_graph.query.return_value = _make_result(["id"], [["456"]])
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            result = await backend.execute_query(
                "CREATE (n:Memory {id: $id}) RETURN n.id as id",
                parameters={"id": "456"},
                write=True
            )

            assert len(result) == 1
            assert result[0]["id"] == "456"

    @pytest.mark.asyncio
    async def test_execute_query_not_connected(self):
        backend = FalkorDBBackend(host='localhost', port=6379)

        with pytest.raises(DatabaseConnectionError, match="(?i)not connected"):
            await backend.execute_query("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_execute_query_dict_row_bypasses_convert(self):
        """Dict rows in result_set are passed through without _convert_falkordb_value."""
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

            mock_result = Mock()
            mock_result.header = [[1, "id"]]
            mock_result.result_set = [{"id": "789"}]
            mock_graph.query.return_value = mock_result
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            with patch.object(
                FalkorDBBackend, '_convert_falkordb_value', wraps=FalkorDBBackend._convert_falkordb_value
            ) as mock_convert:
                result = await backend.execute_query("RETURN 'test' as id", write=False)

            mock_graph.query.assert_called_once()
            assert "RETURN 'test' as id" in mock_graph.query.call_args[0][0]
            assert result == [{"id": "789"}]
            mock_convert.assert_not_called()


class TestFalkorDBSchema:

    @pytest.mark.asyncio
    async def test_initialize_schema(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            backend.execute_query = AsyncMock()
            await backend.initialize_schema()

            assert backend.execute_query.call_count >= 2


class TestFalkorDBMemoryOperations:

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
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

            mock_graph.query.return_value = _make_result(["id"], [[sample_memory.id]])
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            memory_id = await backend.store_memory(sample_memory)

            assert memory_id == sample_memory.id

    @pytest.mark.asyncio
    async def test_get_memory(self, sample_memory):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

            node = _make_node({
                "id": sample_memory.id,
                "type": "solution",
                "title": "Redis Timeout Fix",
                "content": "Increased connection timeout to 5000ms",
                "summary": None,
                "tags": ["redis", "timeout", "performance"],
                "importance": 0.8,
                "confidence": 0.9,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "usage_count": 0
            })
            mock_graph.query.return_value = _make_result(["m"], [[node]])
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            memory = await backend.get_memory(sample_memory.id)

            assert memory is not None
            assert memory.id == sample_memory.id
            assert memory.title == "Redis Timeout Fix"

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()
            mock_result = Mock()
            mock_result.result_set = []
            mock_result.header = []
            mock_graph.query.return_value = mock_result
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            memory = await backend.get_memory("nonexistent")

            assert memory is None

    @pytest.mark.asyncio
    async def test_update_memory(self, sample_memory):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

            mock_graph.query.return_value = _make_result(["id"], [[sample_memory.id]])
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            sample_memory.title = "Updated Title"
            result = await backend.update_memory(sample_memory)

            assert result is True

    @pytest.mark.asyncio
    async def test_delete_memory(self, sample_memory):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

            exists_result = _make_result(["id"], [[sample_memory.id]])
            delete_result = Mock()
            delete_result.result_set = []
            delete_result.header = []
            mock_graph.query.side_effect = [exists_result, delete_result]
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            result = await backend.delete_memory(sample_memory.id)

            assert result is True


class TestFalkorDBRelationships:

    @pytest.mark.asyncio
    async def test_create_relationship(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()
            rel_id = str(uuid.uuid4())

            mock_graph.query.return_value = _make_result(["id"], [[rel_id]])
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

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
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

            related_node = _make_node({
                "id": "mem2",
                "type": "solution",
                "title": "Related Memory",
                "content": "Content",
                "tags": [],
                "importance": 0.7,
                "confidence": 0.8,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "usage_count": 0
            })
            rel_props = {
                "strength": 0.9,
                "confidence": 0.8,
                "context": "Test context"
            }
            mock_graph.query.return_value = _make_result(
                ["related", "rel_type", "rel_props"],
                [[related_node, "SOLVES", rel_props]]
            )
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            related = await backend.get_related_memories("mem1")

            assert len(related) == 1
            memory, relationship = related[0]
            assert memory.id == "mem2"
            assert relationship.type == RelationshipType.SOLVES


class TestFalkorDBSearch:

    @pytest.mark.asyncio
    async def test_search_memories(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

            node = _make_node({
                "id": "search1",
                "type": "solution",
                "title": "Redis Timeout",
                "content": "Fix for timeout",
                "tags": ["redis"],
                "importance": 0.8,
                "confidence": 0.9,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "usage_count": 0
            })
            mock_graph.query.return_value = _make_result(["m"], [[node]])
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            results = await backend.search_memories(SearchQuery(query="timeout"))

            assert len(results) == 1
            assert results[0].id == "search1"
            assert results[0].title == "Redis Timeout"


class TestFalkorDBStatistics:

    @pytest.mark.asyncio
    async def test_get_memory_statistics(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

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

            mock_graph.query.side_effect = mock_query_side_effect
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            stats = await backend.get_memory_statistics()

            assert "total_memories" in stats
            assert "memories_by_type" in stats
            assert stats["memories_by_type"]["solution"] == 20


class TestFalkorDBHealthCheck:

    @pytest.mark.asyncio
    async def test_health_check_connected(self):
        with patch('falkordb.FalkorDB') as mock_falkordb_class:
            mock_client = Mock()
            mock_graph = Mock()

            mock_graph.query.return_value = _make_result(["count"], [[10]])
            mock_client.select_graph.return_value = mock_graph
            mock_falkordb_class.return_value = mock_client

            backend = FalkorDBBackend(host='localhost', port=6379)
            await backend.connect()

            health = await backend.health_check()

            assert health["connected"] is True
            assert health["backend_type"] == "falkordb"
            assert health["statistics"]["memory_count"] == 10

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self):
        backend = FalkorDBBackend(host='localhost', port=6379)

        health = await backend.health_check()

        assert health["connected"] is False
        assert health["backend_type"] == "falkordb"


class TestConvertFalkorDBValue:

    def test_node_conversion(self):
        node = _make_node({"id": "123", "title": "Test"})
        result = FalkorDBBackend._convert_falkordb_value(node)
        assert result == {"id": "123", "title": "Test"}

    def test_scalar_passthrough(self):
        assert FalkorDBBackend._convert_falkordb_value("hello") == "hello"
        assert FalkorDBBackend._convert_falkordb_value(42) == 42
        assert FalkorDBBackend._convert_falkordb_value(None) is None

    def test_dict_passthrough(self):
        d = {"key": "value"}
        assert FalkorDBBackend._convert_falkordb_value(d) == d
