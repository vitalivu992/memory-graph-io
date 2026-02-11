"""Coverage tests for FalkorDB backend -- error handling and edge cases."""

from unittest.mock import Mock, patch

import pytest

from memorygraph.backends.falkordb_backend import FalkorDBBackend
from memorygraph.config import Config
from memorygraph.models import DatabaseConnectionError


class TestFalkorDBQueryErrors:

    @pytest.mark.asyncio
    async def test_execute_query_with_exception(self):
        backend = FalkorDBBackend(host="localhost", port=6379)

        mock_graph = Mock()
        mock_graph.query.side_effect = Exception("Database error")

        backend.client = Mock()
        backend.graph = mock_graph
        backend._connected = True

        with pytest.raises(DatabaseConnectionError, match="Query execution failed"):
            await backend.execute_query("MATCH (n) RETURN n")


class TestFalkorDBSchemaErrorHandling:

    @pytest.mark.asyncio
    async def test_initialize_schema_constraint_errors(self):
        """Schema initialization continues on constraint errors."""
        backend = FalkorDBBackend(host="localhost", port=6379)
        call_count = [0]

        async def mock_execute_query(query, parameters=None, write=False):
            call_count[0] += 1
            if "CONSTRAINT" in query:
                raise Exception("Constraint already exists")
            return []

        backend.execute_query = mock_execute_query
        backend._connected = True

        await backend.initialize_schema()
        assert call_count[0] > 0

    @pytest.mark.asyncio
    async def test_initialize_schema_index_errors(self):
        """Schema initialization continues on index errors."""
        backend = FalkorDBBackend(host="localhost", port=6379)
        call_count = [0]

        async def mock_execute_query(query, parameters=None, write=False):
            call_count[0] += 1
            if "INDEX" in query and "CREATE INDEX" in query:
                raise Exception("Index already exists")
            return []

        backend.execute_query = mock_execute_query
        backend._connected = True

        await backend.initialize_schema()
        assert call_count[0] > 0


class TestFalkorDBMultiTenantMode:

    @pytest.mark.asyncio
    async def test_initialize_schema_with_multitenant_mode(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        queries_executed = []

        async def mock_execute_query(query, parameters=None, write=False):
            queries_executed.append(query)
            return []

        backend.execute_query = mock_execute_query
        backend._connected = True

        with patch.object(Config, 'is_multi_tenant_mode', return_value=True):
            await backend.initialize_schema()

        index_queries = [q for q in queries_executed if "CREATE INDEX" in q]
        tenant_indexes = [q for q in index_queries if any(
            field in q for field in ['context_tenant_id', 'context_team_id', 'context_visibility']
        )]
        assert len(tenant_indexes) >= 3

    @pytest.mark.asyncio
    async def test_initialize_schema_without_multitenant_mode(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        queries_executed = []

        async def mock_execute_query(query, parameters=None, write=False):
            queries_executed.append(query)
            return []

        backend.execute_query = mock_execute_query
        backend._connected = True

        with patch.object(Config, 'is_multi_tenant_mode', return_value=False):
            await backend.initialize_schema()

        tenant_indexes = [q for q in queries_executed if 'context_tenant_id' in q]
        assert len(tenant_indexes) == 0


class TestFalkorDBConnectionEdgeCases:

    def test_backend_initialization_with_defaults(self):
        backend = FalkorDBBackend()

        assert backend.host == Config.FALKORDB_HOST
        assert backend.port == Config.FALKORDB_PORT
        assert backend.password == Config.FALKORDB_PASSWORD

    def test_backend_initialization_with_config(self):
        with patch.object(Config, 'FALKORDB_HOST', "custom-host"):
            with patch.object(Config, 'FALKORDB_PORT', 7000):
                with patch.object(Config, 'FALKORDB_PASSWORD', "secret"):
                    backend = FalkorDBBackend()

                    assert backend.host == "custom-host"
                    assert backend.port == 7000
                    assert backend.password == "secret"

    def test_backend_initialization_with_explicit_params(self):
        backend = FalkorDBBackend(host="explicit-host", port=8000, password="explicit-pass")

        assert backend.host == "explicit-host"
        assert backend.port == 8000
        assert backend.password == "explicit-pass"


class TestFalkorDBContextManagerAndCleanup:

    @pytest.mark.asyncio
    async def test_disconnect_with_client(self):
        backend = FalkorDBBackend(host="localhost", port=6379)

        backend.client = Mock()
        backend._connected = True

        await backend.disconnect()

        assert backend.client is None
        assert backend.graph is None
        assert backend._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_without_client(self):
        backend = FalkorDBBackend(host="localhost", port=6379)
        backend.client = None
        backend._connected = False

        await backend.disconnect()

        assert backend._connected is False
