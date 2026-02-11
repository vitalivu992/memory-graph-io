"""Coverage tests for FalkorDBLite backend -- error handling and edge cases."""

from unittest.mock import Mock, patch

import pytest

from memorygraph.backends.falkordblite_backend import FalkorDBLiteBackend
from memorygraph.config import Config
from memorygraph.models import DatabaseConnectionError


class TestFalkorDBLiteQueryErrors:

    @pytest.mark.asyncio
    async def test_execute_query_with_exception(self):
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")

        mock_graph = Mock()
        mock_graph.query.side_effect = Exception("Database error")

        backend.client = Mock()
        backend.graph = mock_graph
        backend._connected = True

        with pytest.raises(DatabaseConnectionError, match="Query execution failed"):
            await backend.execute_query("MATCH (n) RETURN n")


class TestFalkorDBLiteSchemaErrorHandling:

    @pytest.mark.asyncio
    async def test_initialize_schema_constraint_errors(self):
        """Schema initialization continues on constraint errors."""
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")
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
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")
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


class TestFalkorDBLiteMultiTenantMode:

    @pytest.mark.asyncio
    async def test_initialize_schema_with_multitenant_mode(self):
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")
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
            field in q for field in ['context_tenant_id', 'context_team_id', 'context_visibility', 'context_created_by', 'version']
        )]
        assert len(tenant_indexes) >= 5

    @pytest.mark.asyncio
    async def test_initialize_schema_without_multitenant_mode(self):
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")
        queries_executed = []

        async def mock_execute_query(query, parameters=None, write=False):
            queries_executed.append(query)
            return []

        backend.execute_query = mock_execute_query
        backend._connected = True

        with patch.object(Config, 'is_multi_tenant_mode', return_value=False):
            await backend.initialize_schema()

        tenant_indexes = [q for q in queries_executed if any(
            field in q for field in ['context_tenant_id', 'context_team_id']
        )]
        assert len(tenant_indexes) == 0


class TestFalkorDBLiteConnectionEdgeCases:

    def test_backend_initialization_with_default_path(self):
        backend = FalkorDBLiteBackend()

        assert backend.db_path == Config.FALKORDBLITE_PATH
        assert '.memorygraph' in backend.db_path
        assert 'falkordblite.db' in backend.db_path

    def test_backend_initialization_with_config_path(self):
        test_path = "/custom/path/db.rdb"
        with patch.object(Config, 'FALKORDBLITE_PATH', test_path):
            backend = FalkorDBLiteBackend()
            assert backend.db_path == test_path

    def test_backend_initialization_with_explicit_path(self):
        test_path = "/explicit/path/db.rdb"
        backend = FalkorDBLiteBackend(db_path=test_path)
        assert backend.db_path == test_path


class TestFalkorDBLiteContextManagerAndCleanup:

    @pytest.mark.asyncio
    async def test_disconnect_with_client(self):
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")

        backend.client = Mock()
        backend._connected = True

        await backend.disconnect()

        assert backend.client is None
        assert backend.graph is None
        assert backend._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_without_client(self):
        backend = FalkorDBLiteBackend(db_path="/tmp/test.db")
        backend.client = None
        backend._connected = False

        await backend.disconnect()

        assert backend._connected is False
