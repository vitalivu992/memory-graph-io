"""Tests for the shared FalkorDB base class (_falkordb_shared.py)."""

import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from memorygraph.backends._falkordb_shared import BaseFalkorDBBackend
from memorygraph.backends.falkordb_backend import FalkorDBBackend
from memorygraph.backends.falkordblite_backend import FalkorDBLiteBackend
from memorygraph.config import Config
from memorygraph.models import (
    DatabaseConnectionError,
    Memory,
    MemoryType,
    RelationshipError,
    RelationshipType,
    SearchQuery,
    ValidationError,
)
from tests.backends.conftest import make_connected_backend
from tests.backends.conftest import make_falkordb_node as _make_node
from tests.backends.conftest import make_falkordb_result as _make_result


class TestBaseFalkorDBInheritance:

    def test_falkordb_inherits_from_base(self):
        assert issubclass(FalkorDBBackend, BaseFalkorDBBackend)

    def test_falkordblite_inherits_from_base(self):
        assert issubclass(FalkorDBLiteBackend, BaseFalkorDBBackend)

    def test_display_name_falkordb(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        assert backend._display_name == "FalkorDB"

    def test_display_name_falkordblite(self):
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")
        assert backend._display_name == "FalkorDBLite"


class TestBackwardCompatAliases:

    def test_falkordb_to_memory_alias_exists(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        assert hasattr(backend, "_falkordb_to_memory")
        assert backend._falkordb_to_memory == backend._node_to_memory

    def test_falkordblite_to_memory_alias_exists(self):
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")
        assert hasattr(backend, "_falkordblite_to_memory")
        assert backend._falkordblite_to_memory == backend._node_to_memory


class TestSharedDisconnect:

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        assert backend._connected is True

        await backend.disconnect()

        assert backend.client is None
        assert backend.graph is None
        assert backend._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        backend.client = None
        backend._connected = False

        await backend.disconnect()
        assert backend._connected is False


class TestSharedExecuteQuery:

    @pytest.mark.asyncio
    async def test_not_connected_raises(self):
        backend = FalkorDBBackend(host="localhost", port=6379)

        with pytest.raises(DatabaseConnectionError, match="not connected"):
            await backend.execute_query("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_query_exception_wraps_in_connection_error(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        backend.graph.query.side_effect = RuntimeError("DB crashed")

        with pytest.raises(DatabaseConnectionError, match="Query execution failed"):
            await backend.execute_query("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_empty_result_set(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        assert await backend.execute_query("MATCH (n) RETURN n") == []

    @pytest.mark.asyncio
    async def test_no_result_set_attr(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        backend.graph.query.return_value = Mock(spec=[])

        assert await backend.execute_query("MATCH (n) RETURN n") == []

    @pytest.mark.asyncio
    async def test_result_with_node(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        node = _make_node({"id": "abc", "title": "Test"})
        backend.graph.query.return_value = _make_result(["m"], [[node]])

        records = await backend.execute_query("MATCH (m) RETURN m")
        assert len(records) == 1
        assert records[0]["m"] == {"id": "abc", "title": "Test"}

    @pytest.mark.asyncio
    async def test_result_with_scalar(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        backend.graph.query.return_value = _make_result(["count"], [[42]])

        records = await backend.execute_query("RETURN 42 as count")
        assert records[0]["count"] == 42

    @pytest.mark.asyncio
    async def test_result_with_dict_row(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        result = Mock()
        result.header = [[1, "id"]]
        result.result_set = [{"id": "xyz"}]
        backend.graph.query.return_value = result

        records = await backend.execute_query("RETURN 'xyz' as id")
        assert records[0]["id"] == "xyz"

    @pytest.mark.asyncio
    async def test_header_with_plain_string(self):
        """Plain string headers (not [type, name] pairs) are handled correctly."""
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        result = Mock()
        result.header = ["col1"]
        result.result_set = [["val1"]]
        backend.graph.query.return_value = result

        records = await backend.execute_query("RETURN 'val1'")
        assert records[0]["col1"] == "val1"


class TestSharedCRUD:

    @pytest.fixture
    def memory(self):
        return Memory(
            id=str(uuid.uuid4()),
            type=MemoryType.SOLUTION,
            title="Test Memory",
            content="Test content",
            tags=["test"],
            importance=0.8,
            confidence=0.9,
        )

    @pytest.mark.asyncio
    async def test_store_memory_assigns_id_if_missing(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        generated_id = str(uuid.uuid4())
        backend.graph.query.return_value = _make_result(["id"], [[generated_id]])

        memory = Memory(
            type=MemoryType.PROBLEM,
            title="No ID",
            content="Memory without an ID",
        )
        memory.id = None

        await backend.store_memory(memory)
        assert memory.id is not None

    @pytest.mark.asyncio
    async def test_store_memory_empty_result_raises(self, memory):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        with pytest.raises(DatabaseConnectionError, match="Failed to store memory"):
            await backend.store_memory(memory)

    @pytest.mark.asyncio
    async def test_get_memory_returns_none_when_not_found(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        assert await backend.get_memory("nonexistent") is None

    @pytest.mark.asyncio
    async def test_update_memory_without_id_raises(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        memory = Memory(
            type=MemoryType.PROBLEM,
            title="No ID",
            content="content",
        )
        memory.id = None

        with pytest.raises(ValidationError, match="Memory must have an ID"):
            await backend.update_memory(memory)

    @pytest.mark.asyncio
    async def test_delete_memory_returns_false_when_not_found(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        result_empty = Mock()
        result_empty.result_set = []
        result_empty.header = []
        backend.graph.query.return_value = result_empty

        assert await backend.delete_memory("nonexistent") is False
        assert backend.graph.query.call_count == 1

    @pytest.mark.asyncio
    async def test_delete_memory_returns_true_when_found(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)

        exists_result = _make_result(["id"], [["mem-123"]])
        delete_result = Mock()
        delete_result.result_set = []
        delete_result.header = []
        backend.graph.query.side_effect = [exists_result, delete_result]

        assert await backend.delete_memory("mem-123") is True
        assert backend.graph.query.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_memory_no_count_after_detach_delete(self):
        """Delete query must use DETACH DELETE without COUNT."""
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)

        exists_result = _make_result(["id"], [["mem-123"]])
        delete_result = Mock()
        delete_result.result_set = []
        delete_result.header = []
        backend.graph.query.side_effect = [exists_result, delete_result]

        await backend.delete_memory("mem-123")

        delete_query = backend.graph.query.call_args_list[1][0][0]
        assert "COUNT" not in delete_query
        assert "DETACH DELETE" in delete_query


class TestSharedRelationships:

    @pytest.mark.asyncio
    async def test_create_relationship_empty_result_raises(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        with pytest.raises(RelationshipError, match="Failed to create relationship"):
            await backend.create_relationship(
                from_memory_id="m1",
                to_memory_id="m2",
                relationship_type=RelationshipType.SOLVES,
            )

    @pytest.mark.asyncio
    async def test_create_relationship_default_properties(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        rel_id = str(uuid.uuid4())
        backend.graph.query.return_value = _make_result(["id"], [[rel_id]])

        result = await backend.create_relationship(
            from_memory_id="m1",
            to_memory_id="m2",
            relationship_type=RelationshipType.RELATED_TO,
        )
        assert result == rel_id

    @pytest.mark.asyncio
    async def test_get_related_memories_empty(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        assert await backend.get_related_memories("mem1") == []

    @pytest.mark.asyncio
    async def test_get_related_memories_with_type_filter(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        node = _make_node({
            "id": "mem2",
            "type": "solution",
            "title": "Related",
            "content": "Content",
            "tags": [],
            "importance": 0.5,
            "confidence": 0.5,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "usage_count": 0,
        })
        backend.graph.query.return_value = _make_result(
            ["related", "rel_type", "rel_props"],
            [[node, "SOLVES", {"strength": 0.9, "confidence": 0.8}]],
        )

        related = await backend.get_related_memories(
            "mem1", relationship_types=[RelationshipType.SOLVES]
        )
        assert len(related) == 1

    @pytest.mark.asyncio
    async def test_get_related_memories_invalid_rel_type_defaults(self):
        """Unknown relationship type defaults to RELATED_TO."""
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        node = _make_node({
            "id": "mem2",
            "type": "solution",
            "title": "Related",
            "content": "Content",
            "tags": [],
            "importance": 0.5,
            "confidence": 0.5,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "usage_count": 0,
        })
        backend.graph.query.return_value = _make_result(
            ["related", "rel_type", "rel_props"],
            [[node, "UNKNOWN_TYPE_XYZ", {}]],
        )

        related = await backend.get_related_memories("mem1")
        assert len(related) == 1
        _, relationship = related[0]
        assert relationship.type == RelationshipType.RELATED_TO


class TestSharedSearch:

    @pytest.mark.asyncio
    async def test_search_with_all_filters(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        node = _make_node({
            "id": "s1",
            "type": "solution",
            "title": "Redis Fix",
            "content": "Fix content",
            "tags": ["redis"],
            "importance": 0.9,
            "confidence": 0.9,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "usage_count": 0,
        })
        backend.graph.query.return_value = _make_result(["m"], [[node]])

        query = SearchQuery(
            query="redis",
            memory_types=[MemoryType.SOLUTION],
            tags=["redis"],
            project_path="/project",
            min_importance=0.5,
            min_confidence=0.5,
        )
        assert len(await backend.search_memories(query)) == 1

    @pytest.mark.asyncio
    async def test_search_no_filters(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        assert await backend.search_memories(SearchQuery()) == []


class TestSharedStatistics:

    @pytest.mark.asyncio
    async def test_statistics_handles_query_failure(self):
        """Statistics should return None values on query failure."""
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        backend.graph.query.side_effect = Exception("DB error")

        stats = await backend.get_memory_statistics()

        assert stats["total_memories"] is None
        assert stats["memories_by_type"] is None


class TestSharedCapabilities:

    def test_supports_fulltext_search(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        assert backend.supports_fulltext_search() is True

    def test_supports_transactions(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        assert backend.supports_transactions() is True

    def test_is_cypher_capable(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        assert backend.is_cypher_capable() is True

    def test_capabilities_consistent_between_backends(self):
        fb = FalkorDBBackend(host="localhost", port=6379)
        fbl = FalkorDBLiteBackend(db_path="/tmp/test.db")

        assert fb.supports_fulltext_search() == fbl.supports_fulltext_search()
        assert fb.supports_transactions() == fbl.supports_transactions()
        assert fb.is_cypher_capable() == fbl.is_cypher_capable()


class TestConvertFalkorDBValue:

    def test_node_to_dict(self):
        node = _make_node({"id": "1", "name": "test"})
        assert BaseFalkorDBBackend._convert_falkordb_value(node) == {"id": "1", "name": "test"}

    def test_scalar_passthrough(self):
        assert BaseFalkorDBBackend._convert_falkordb_value("hello") == "hello"
        assert BaseFalkorDBBackend._convert_falkordb_value(42) == 42
        assert BaseFalkorDBBackend._convert_falkordb_value(None) is None

    def test_dict_passthrough(self):
        d = {"key": "value"}
        assert BaseFalkorDBBackend._convert_falkordb_value(d) == d

    def test_accessible_from_subclasses(self):
        node = _make_node({"id": "1"})
        assert FalkorDBBackend._convert_falkordb_value(node) == {"id": "1"}
        assert FalkorDBLiteBackend._convert_falkordb_value(node) == {"id": "1"}


class TestNodeToMemory:

    def test_valid_node_data(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        node_data = {
            "id": "test-id",
            "type": "solution",
            "title": "Test Title",
            "content": "Test content",
            "tags": ["test"],
            "importance": 0.8,
            "confidence": 0.9,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "usage_count": 0,
        }
        memory = backend._node_to_memory(node_data)
        assert memory is not None
        assert memory.id == "test-id"
        assert memory.title == "Test Title"

    def test_alias_produces_same_result(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        node_data = {
            "id": "test-id",
            "type": "solution",
            "title": "Alias Test",
            "content": "content",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        m1 = backend._node_to_memory(node_data)
        m2 = backend._falkordb_to_memory(node_data)
        assert m1.id == m2.id
        assert m1.title == m2.title

    def test_falkordblite_alias_produces_same_result(self):
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")
        node_data = {
            "id": "test-id",
            "type": "problem",
            "title": "Lite Alias Test",
            "content": "content",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        m1 = backend._node_to_memory(node_data)
        m2 = backend._falkordblite_to_memory(node_data)
        assert m1.id == m2.id
        assert m1.title == m2.title


class TestSharedSchemaMultitenant:

    @pytest.mark.asyncio
    async def test_multitenant_indexes_created(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        queries = []

        async def mock_execute(query, parameters=None, write=False):
            queries.append(query)
            return []

        backend.execute_query = mock_execute

        with patch.object(Config, "is_multi_tenant_mode", return_value=True):
            await backend.initialize_schema()

        tenant_queries = [
            q for q in queries
            if "context_tenant_id" in q or "context_team_id" in q
        ]
        assert len(tenant_queries) >= 2

    @pytest.mark.asyncio
    async def test_no_multitenant_indexes_without_mode(self):
        backend = make_connected_backend(FalkorDBBackend, host="localhost", port=6379)
        queries = []

        async def mock_execute(query, parameters=None, write=False):
            queries.append(query)
            return []

        backend.execute_query = mock_execute

        with patch.object(Config, "is_multi_tenant_mode", return_value=False):
            await backend.initialize_schema()

        assert not any("context_tenant_id" in q for q in queries)
