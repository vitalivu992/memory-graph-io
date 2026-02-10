"""
Tests for the shared FalkorDB base class (_falkordb_shared.py).

These tests verify the shared logic that both FalkorDBBackend and
FalkorDBLiteBackend inherit from BaseFalkorDBBackend, ensuring the
refactored code maintains correct behavior.
"""

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


def _make_node(properties: dict) -> Mock:
    """Create a mock FalkorDB Node with a properties dict."""
    node = Mock()
    node.properties = properties
    return node


def _make_result(header_names: list, rows: list) -> Mock:
    """Create a mock FalkorDB QueryResult matching the real format."""
    result = Mock()
    result.header = [[1, name] for name in header_names]
    result.result_set = rows
    return result


def _make_connected_backend(cls=FalkorDBBackend, **kwargs):
    """Create a backend with mocked connection state."""
    if cls == FalkorDBBackend:
        backend = cls(host="localhost", port=6379, **kwargs)
    else:
        backend = cls(db_path="/tmp/test.db", **kwargs)
    backend.client = Mock()
    backend.graph = Mock()
    backend._connected = True
    return backend


class TestBaseFalkorDBInheritance:
    """Verify both subclasses correctly inherit from the shared base."""

    def test_falkordb_inherits_from_base(self):
        """FalkorDBBackend should inherit from BaseFalkorDBBackend."""
        assert issubclass(FalkorDBBackend, BaseFalkorDBBackend)

    def test_falkordblite_inherits_from_base(self):
        """FalkorDBLiteBackend should inherit from BaseFalkorDBBackend."""
        assert issubclass(FalkorDBLiteBackend, BaseFalkorDBBackend)

    def test_display_name_falkordb(self):
        """FalkorDBBackend should have correct display name."""
        backend = FalkorDBBackend(host="localhost", port=6379)
        assert backend._display_name == "FalkorDB"

    def test_display_name_falkordblite(self):
        """FalkorDBLiteBackend should have correct display name."""
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")
        assert backend._display_name == "FalkorDBLite"


class TestBackwardCompatAliases:
    """Verify backward-compatible method aliases work."""

    def test_falkordb_to_memory_alias_exists(self):
        """FalkorDBBackend should have _falkordb_to_memory alias."""
        backend = FalkorDBBackend(host="localhost", port=6379)
        assert hasattr(backend, "_falkordb_to_memory")
        assert backend._falkordb_to_memory == backend._node_to_memory

    def test_falkordblite_to_memory_alias_exists(self):
        """FalkorDBLiteBackend should have _falkordblite_to_memory alias."""
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")
        assert hasattr(backend, "_falkordblite_to_memory")
        assert backend._falkordblite_to_memory == backend._node_to_memory


class TestSharedDisconnect:
    """Test shared disconnect behavior."""

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self):
        """Disconnect should clear client, graph, and connected flag."""
        backend = _make_connected_backend(FalkorDBBackend)
        assert backend._connected is True

        await backend.disconnect()

        assert backend.client is None
        assert backend.graph is None
        assert backend._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self):
        """Disconnect should be safe to call when already disconnected."""
        backend = FalkorDBBackend(host="localhost", port=6379)
        backend.client = None
        backend._connected = False

        await backend.disconnect()
        assert backend._connected is False


class TestSharedExecuteQuery:
    """Test shared query execution logic."""

    @pytest.mark.asyncio
    async def test_not_connected_raises(self):
        """Should raise DatabaseConnectionError when not connected."""
        backend = FalkorDBBackend(host="localhost", port=6379)

        with pytest.raises(DatabaseConnectionError, match="not connected"):
            await backend.execute_query("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_query_exception_wraps_in_connection_error(self):
        """Query exceptions should be wrapped in DatabaseConnectionError."""
        backend = _make_connected_backend(FalkorDBBackend)
        backend.graph.query.side_effect = RuntimeError("DB crashed")

        with pytest.raises(DatabaseConnectionError, match="Query execution failed"):
            await backend.execute_query("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_empty_result_set(self):
        """Empty result set should return empty list."""
        backend = _make_connected_backend(FalkorDBBackend)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        records = await backend.execute_query("MATCH (n) RETURN n")
        assert records == []

    @pytest.mark.asyncio
    async def test_no_result_set_attr(self):
        """No result_set attribute should return empty list."""
        backend = _make_connected_backend(FalkorDBBackend)
        result = Mock(spec=[])  # no attributes at all
        backend.graph.query.return_value = result

        records = await backend.execute_query("MATCH (n) RETURN n")
        assert records == []

    @pytest.mark.asyncio
    async def test_result_with_node(self):
        """Node objects should be converted to property dicts."""
        backend = _make_connected_backend(FalkorDBBackend)
        node = _make_node({"id": "abc", "title": "Test"})
        backend.graph.query.return_value = _make_result(["m"], [[node]])

        records = await backend.execute_query("MATCH (m) RETURN m")
        assert len(records) == 1
        assert records[0]["m"] == {"id": "abc", "title": "Test"}

    @pytest.mark.asyncio
    async def test_result_with_scalar(self):
        """Scalar values should pass through unchanged."""
        backend = _make_connected_backend(FalkorDBBackend)
        backend.graph.query.return_value = _make_result(["count"], [[42]])

        records = await backend.execute_query("RETURN 42 as count")
        assert records[0]["count"] == 42

    @pytest.mark.asyncio
    async def test_result_with_dict_row(self):
        """Dict rows should pass through unchanged."""
        backend = _make_connected_backend(FalkorDBBackend)
        result = Mock()
        result.header = [[1, "id"]]
        result.result_set = [{"id": "xyz"}]
        backend.graph.query.return_value = result

        records = await backend.execute_query("RETURN 'xyz' as id")
        assert records[0]["id"] == "xyz"

    @pytest.mark.asyncio
    async def test_header_with_plain_string(self):
        """Headers that are plain strings (not [type, name]) should be handled."""
        backend = _make_connected_backend(FalkorDBBackend)
        result = Mock()
        result.header = ["col1"]  # plain string, not [type, name]
        result.result_set = [["val1"]]
        backend.graph.query.return_value = result

        records = await backend.execute_query("RETURN 'val1'")
        assert records[0]["col1"] == "val1"


class TestSharedCRUD:
    """Test shared CRUD operations on the base class."""

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
        """store_memory should assign an ID if the memory doesn't have one."""
        backend = _make_connected_backend(FalkorDBBackend)
        generated_id = str(uuid.uuid4())
        backend.graph.query.return_value = _make_result(["id"], [[generated_id]])

        memory = Memory(
            type=MemoryType.PROBLEM,
            title="No ID",
            content="Memory without an ID",
        )
        memory.id = None

        await backend.store_memory(memory)
        # The memory should now have an ID assigned
        assert memory.id is not None

    @pytest.mark.asyncio
    async def test_store_memory_empty_result_raises(self, memory):
        """store_memory should raise if no result returned."""
        backend = _make_connected_backend(FalkorDBBackend)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        with pytest.raises(DatabaseConnectionError, match="Failed to store memory"):
            await backend.store_memory(memory)

    @pytest.mark.asyncio
    async def test_get_memory_returns_none_when_not_found(self):
        """get_memory should return None for non-existent ID."""
        backend = _make_connected_backend(FalkorDBBackend)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        assert await backend.get_memory("nonexistent") is None

    @pytest.mark.asyncio
    async def test_update_memory_without_id_raises(self):
        """update_memory should raise ValidationError if no ID."""
        backend = _make_connected_backend(FalkorDBBackend)
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
        """delete_memory should return False if memory doesn't exist."""
        backend = _make_connected_backend(FalkorDBBackend)
        backend.graph.query.return_value = _make_result(["deleted_count"], [[0]])

        result = await backend.delete_memory("nonexistent")
        assert result is False


class TestSharedRelationships:
    """Test shared relationship operations."""

    @pytest.mark.asyncio
    async def test_create_relationship_empty_result_raises(self):
        """create_relationship should raise RelationshipError on empty result."""
        backend = _make_connected_backend(FalkorDBBackend)
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
        """create_relationship should use default properties when none given."""
        backend = _make_connected_backend(FalkorDBBackend)
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
        """get_related_memories should return empty list when no relations."""
        backend = _make_connected_backend(FalkorDBBackend)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        related = await backend.get_related_memories("mem1")
        assert related == []

    @pytest.mark.asyncio
    async def test_get_related_memories_with_type_filter(self):
        """get_related_memories should filter by relationship type."""
        backend = _make_connected_backend(FalkorDBBackend)
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
        """Unknown relationship type in result should default to RELATED_TO."""
        backend = _make_connected_backend(FalkorDBBackend)
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
    """Test shared search logic."""

    @pytest.mark.asyncio
    async def test_search_with_all_filters(self):
        """search_memories should handle all filter criteria."""
        backend = _make_connected_backend(FalkorDBBackend)
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
        results = await backend.search_memories(query)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_no_filters(self):
        """search_memories should work with no filters (just limit)."""
        backend = _make_connected_backend(FalkorDBBackend)
        result = Mock()
        result.result_set = []
        result.header = []
        backend.graph.query.return_value = result

        query = SearchQuery()
        results = await backend.search_memories(query)
        assert results == []


class TestSharedStatistics:
    """Test shared statistics logic."""

    @pytest.mark.asyncio
    async def test_statistics_handles_query_failure(self):
        """Statistics should handle individual query failures gracefully."""
        backend = _make_connected_backend(FalkorDBBackend)
        backend.graph.query.side_effect = Exception("DB error")

        stats = await backend.get_memory_statistics()

        # All stat keys should be present but None
        assert "total_memories" in stats
        assert stats["total_memories"] is None
        assert "memories_by_type" in stats
        assert stats["memories_by_type"] is None


class TestSharedCapabilities:
    """Test shared capability flag methods."""

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
        """Both backends should report the same capabilities."""
        fb = FalkorDBBackend(host="localhost", port=6379)
        fbl = FalkorDBLiteBackend(db_path="/tmp/test.db")

        assert fb.supports_fulltext_search() == fbl.supports_fulltext_search()
        assert fb.supports_transactions() == fbl.supports_transactions()
        assert fb.is_cypher_capable() == fbl.is_cypher_capable()


class TestConvertFalkorDBValue:
    """Test the shared _convert_falkordb_value static method."""

    def test_node_to_dict(self):
        node = _make_node({"id": "1", "name": "test"})
        result = BaseFalkorDBBackend._convert_falkordb_value(node)
        assert result == {"id": "1", "name": "test"}

    def test_scalar_passthrough(self):
        assert BaseFalkorDBBackend._convert_falkordb_value("hello") == "hello"
        assert BaseFalkorDBBackend._convert_falkordb_value(42) == 42
        assert BaseFalkorDBBackend._convert_falkordb_value(None) is None

    def test_dict_passthrough(self):
        d = {"key": "value"}
        assert BaseFalkorDBBackend._convert_falkordb_value(d) == d

    def test_accessible_from_subclasses(self):
        """Static method should be accessible from both subclasses."""
        node = _make_node({"id": "1"})
        assert FalkorDBBackend._convert_falkordb_value(node) == {"id": "1"}
        assert FalkorDBLiteBackend._convert_falkordb_value(node) == {"id": "1"}


class TestNodeToMemory:
    """Test the shared _node_to_memory method."""

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
        """_falkordb_to_memory alias should produce same result as _node_to_memory."""
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
        """_falkordblite_to_memory alias should produce same result as _node_to_memory."""
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
    """Test schema initialization with multi-tenant mode in shared base."""

    @pytest.mark.asyncio
    async def test_multitenant_indexes_created(self):
        """Multi-tenant indexes should be created when mode is enabled."""
        backend = _make_connected_backend(FalkorDBBackend)
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
        """Multi-tenant indexes should NOT be created when mode is disabled."""
        backend = _make_connected_backend(FalkorDBBackend)
        queries = []

        async def mock_execute(query, parameters=None, write=False):
            queries.append(query)
            return []

        backend.execute_query = mock_execute

        with patch.object(Config, "is_multi_tenant_mode", return_value=False):
            await backend.initialize_schema()

        tenant_queries = [q for q in queries if "context_tenant_id" in q]
        assert len(tenant_queries) == 0
