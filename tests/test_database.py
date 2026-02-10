"""
Comprehensive tests for database module.

Tests cover:
- Connection management
- Schema initialization
- CRUD operations
- Relationship management
- Error handling
- Async operations
"""

import asyncio
import importlib.util
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorygraph.database import MemoryDatabase, Neo4jConnection
from memorygraph.models import (
    DatabaseConnectionError,
    Memory,
    MemoryContext,
    MemoryType,
    RelationshipProperties,
    RelationshipType,
    SearchQuery,
)
from tests.conftest import patch_config

neo4j_available = importlib.util.find_spec("neo4j") is not None
neo4j_skip = pytest.mark.skipif(
    not neo4j_available,
    reason="neo4j package not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_mock_execute(return_data):
    """Create an async execute mock that runs the transaction function."""
    async def mock_execute(func, *args):
        mock_tx = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=return_data)
        mock_tx.run = AsyncMock(return_value=mock_result)
        return await func(mock_tx, *args)
    return mock_execute


def _wire_session(mock_driver, mock_session):
    """Connect *mock_driver.session* to return *mock_session*."""
    mock_driver.session = MagicMock(return_value=mock_session)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_driver():
    """Create a mock async Neo4j driver."""
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def mock_session():
    """Create a mock async Neo4j session."""
    session = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
async def connection(mock_driver):
    """Create a test Neo4j connection with the mock driver attached."""
    conn = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="test_password",
    )
    conn.driver = mock_driver
    return conn


@pytest.fixture
async def database(connection):
    """Create a test MemoryDatabase instance."""
    return MemoryDatabase(connection)


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        id=str(uuid.uuid4()),
        type=MemoryType.SOLUTION,
        title="Test Solution",
        content="This is a test solution for a problem",
        tags=["python", "testing"],
        importance=0.8,
        confidence=0.9,
        context=MemoryContext(
            project_path="/test/project",
            files_involved=["test.py"],
            languages=["python"],
        ),
    )


# ---------------------------------------------------------------------------
# Neo4jConnection tests
# ---------------------------------------------------------------------------

class TestNeo4jConnection:
    """Test Neo4j connection management."""

    def test_connection_initialization(self):
        """Test connection initialization with valid credentials."""
        conn = Neo4jConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        assert conn.uri == "bolt://localhost:7687"
        assert conn.user == "neo4j"
        assert conn.password == "password"
        assert conn.driver is None

    def test_connection_initialization_from_config(self):
        """Test connection initialization from Config class."""
        with patch_config(
            NEO4J_URI="bolt://custom:7687",
            NEO4J_USER="custom_user",
            NEO4J_PASSWORD="custom_pass",
        ):
            conn = Neo4jConnection()
            assert conn.uri == "bolt://custom:7687"
            assert conn.user == "custom_user"
            assert conn.password == "custom_pass"

    def test_connection_initialization_missing_password(self):
        """Test that missing password raises error."""
        with pytest.raises(DatabaseConnectionError, match="password must be provided"):
            Neo4jConnection(uri="bolt://localhost:7687", user="neo4j")

    @neo4j_skip
    @pytest.mark.asyncio
    async def test_connect_success(self, connection, mock_driver):
        """Test successful connection to Neo4j."""
        with patch("neo4j.AsyncGraphDatabase") as mock_graph_db:
            mock_graph_db.driver.return_value = mock_driver
            connection.driver = None
            await connection.connect()

            assert connection.driver is not None
            mock_driver.verify_connectivity.assert_called_once()

    @neo4j_skip
    @pytest.mark.asyncio
    async def test_connect_service_unavailable(self):
        """Test connection failure when service is unavailable."""
        from neo4j.exceptions import ServiceUnavailable

        conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", password="password")

        with patch("neo4j.AsyncGraphDatabase") as mock_graph_db:
            driver = AsyncMock()
            driver.verify_connectivity = AsyncMock(side_effect=ServiceUnavailable("Service unavailable"))
            mock_graph_db.driver.return_value = driver

            with pytest.raises(DatabaseConnectionError, match="Failed to connect"):
                await conn.connect()

    @neo4j_skip
    @pytest.mark.asyncio
    async def test_connect_auth_error(self):
        """Test connection failure with authentication error."""
        from neo4j.exceptions import AuthError

        conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", password="wrong")

        with patch("neo4j.AsyncGraphDatabase") as mock_graph_db:
            driver = AsyncMock()
            driver.verify_connectivity = AsyncMock(side_effect=AuthError("Auth failed"))
            mock_graph_db.driver.return_value = driver

            with pytest.raises(DatabaseConnectionError, match="Authentication failed"):
                await conn.connect()

    @neo4j_skip
    @pytest.mark.asyncio
    async def test_connect_unexpected_error(self):
        """Test connection failure with unexpected error."""
        conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", password="password")

        with patch("neo4j.AsyncGraphDatabase") as mock_graph_db:
            driver = AsyncMock()
            driver.verify_connectivity = AsyncMock(side_effect=RuntimeError("Unexpected error"))
            mock_graph_db.driver.return_value = driver

            with pytest.raises(DatabaseConnectionError, match="Unexpected error"):
                await conn.connect()

    @pytest.mark.asyncio
    async def test_connect_neo4j_import_error(self):
        """Test connection failure when neo4j package is not installed."""
        conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", password="password")

        with patch("builtins.__import__", side_effect=ImportError("No module named 'neo4j'")):
            with pytest.raises(DatabaseConnectionError, match="neo4j package is required"):
                await conn.connect()

    @pytest.mark.asyncio
    async def test_close_connection(self, connection, mock_driver):
        """Test closing database connection."""
        await connection.close()
        mock_driver.close.assert_called_once()
        assert connection.driver is None

    @pytest.mark.asyncio
    async def test_session_context_manager(self, connection, mock_driver, mock_session):
        """Test async session context manager."""
        _wire_session(mock_driver, mock_session)

        async with connection.session() as session:
            assert session is not None

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_not_connected(self):
        """Test session creation when not connected."""
        conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", password="password")
        conn.driver = None

        with pytest.raises(DatabaseConnectionError, match="(?i)not connected"):
            async with conn.session():
                pass

    @pytest.mark.asyncio
    async def test_execute_write_query(self, connection, mock_driver, mock_session):
        """Test executing write query."""
        mock_session.execute_write = create_mock_execute([{"created": 1}])
        _wire_session(mock_driver, mock_session)

        result = await connection.execute_write_query(
            "CREATE (n:Test {name: $name}) RETURN n",
            {"name": "test"},
        )

        assert result == [{"created": 1}]

    @pytest.mark.asyncio
    async def test_execute_read_query(self, connection, mock_driver, mock_session):
        """Test executing read query."""
        mock_session.execute_read = create_mock_execute([{"id": "123", "name": "test"}])
        _wire_session(mock_driver, mock_session)

        result = await connection.execute_read_query("MATCH (n:Test) RETURN n", {})

        assert len(result) == 1
        assert result[0]["name"] == "test"

    @neo4j_skip
    @pytest.mark.asyncio
    async def test_execute_write_query_neo4j_error(self, connection, mock_driver, mock_session):
        """Test write query failure with Neo4jError."""
        from neo4j.exceptions import Neo4jError

        async def mock_execute_write_error(func, *args):
            raise Neo4jError("Transaction failed")

        mock_session.execute_write = mock_execute_write_error
        _wire_session(mock_driver, mock_session)

        with pytest.raises(DatabaseConnectionError, match="Write query failed"):
            await connection.execute_write_query("CREATE (n:Test)", {})

    @neo4j_skip
    @pytest.mark.asyncio
    async def test_execute_read_query_neo4j_error(self, connection, mock_driver, mock_session):
        """Test read query failure with Neo4jError."""
        from neo4j.exceptions import Neo4jError

        async def mock_execute_read_error(func, *args):
            raise Neo4jError("Query failed")

        mock_session.execute_read = mock_execute_read_error
        _wire_session(mock_driver, mock_session)

        with pytest.raises(DatabaseConnectionError, match="Read query failed"):
            await connection.execute_read_query("MATCH (n) RETURN n", {})


# ---------------------------------------------------------------------------
# MemoryDatabase tests
# ---------------------------------------------------------------------------

class TestMemoryDatabase:
    """Test MemoryDatabase operations."""

    @pytest.mark.asyncio
    async def test_initialize_schema(self, database, connection, mock_driver, mock_session):
        """Test schema initialization completes without error."""
        mock_session.execute_write = create_mock_execute([])
        _wire_session(mock_driver, mock_session)

        await database.initialize_schema()

    @pytest.mark.asyncio
    async def test_initialize_schema_constraint_exists(self, database, connection, mock_driver, mock_session):
        """Test schema initialization when constraints already exist."""
        call_count = 0

        async def mock_execute_with_exists(func, *args):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception("Constraint already exists")
            mock_tx = AsyncMock()
            mock_result = AsyncMock()
            mock_result.data = AsyncMock(return_value=[])
            mock_tx.run = AsyncMock(return_value=mock_result)
            return await func(mock_tx, *args)

        mock_session.execute_write = mock_execute_with_exists
        _wire_session(mock_driver, mock_session)

        await database.initialize_schema()

    @pytest.mark.asyncio
    async def test_initialize_schema_other_error(self, database, connection, mock_driver, mock_session):
        """Test schema initialization with non-exists errors (logged as warnings)."""
        call_count = 0

        async def mock_execute_with_error(func, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Permission denied")
            mock_tx = AsyncMock()
            mock_result = AsyncMock()
            mock_result.data = AsyncMock(return_value=[])
            mock_tx.run = AsyncMock(return_value=mock_result)
            return await func(mock_tx, *args)

        mock_session.execute_write = mock_execute_with_error
        _wire_session(mock_driver, mock_session)

        await database.initialize_schema()

    @pytest.mark.asyncio
    async def test_store_memory_basic(self, database, connection, sample_memory, mock_driver, mock_session):
        """Test storing a basic memory."""
        mock_session.execute_write = create_mock_execute([{"id": sample_memory.id}])
        _wire_session(mock_driver, mock_session)

        memory_id = await database.store_memory(sample_memory)
        assert memory_id == sample_memory.id

    @pytest.mark.asyncio
    async def test_store_memory_generates_id(self, database, connection, sample_memory, mock_driver, mock_session):
        """Test that store_memory generates ID if not provided."""
        sample_memory.id = None
        generated_id = str(uuid.uuid4())

        mock_session.execute_write = create_mock_execute([{"id": generated_id}])
        _wire_session(mock_driver, mock_session)

        memory_id = await database.store_memory(sample_memory)

        assert memory_id == generated_id
        assert sample_memory.id is not None

    @pytest.mark.asyncio
    async def test_store_memory_no_result(self, database, connection, sample_memory, mock_driver, mock_session):
        """Test store_memory when query returns no result."""
        mock_session.execute_write = create_mock_execute([])
        _wire_session(mock_driver, mock_session)

        with pytest.raises(DatabaseConnectionError, match="Failed to store memory"):
            await database.store_memory(sample_memory)

    @pytest.mark.asyncio
    async def test_store_memory_unexpected_error(self, database, connection, sample_memory, mock_driver, mock_session):
        """Test store_memory with unexpected error."""
        async def mock_execute_error(func, *args):
            raise RuntimeError("Unexpected database error")

        mock_session.execute_write = mock_execute_error
        _wire_session(mock_driver, mock_session)

        with pytest.raises(DatabaseConnectionError, match="Unexpected database error"):
            await database.store_memory(sample_memory)

    @pytest.mark.asyncio
    async def test_get_memory_existing(self, database, connection, sample_memory, mock_driver, mock_session):
        """Test retrieving an existing memory."""
        now = datetime.now(timezone.utc).isoformat()
        memory_data = {
            "m": {
                "id": sample_memory.id,
                "type": "solution",
                "title": sample_memory.title,
                "content": sample_memory.content,
                "tags": sample_memory.tags,
                "importance": sample_memory.importance,
                "confidence": sample_memory.confidence,
                "created_at": now,
                "updated_at": now,
            },
            "relationships": [],
        }
        mock_session.execute_read = create_mock_execute([memory_data])
        _wire_session(mock_driver, mock_session)

        memory = await database.get_memory(sample_memory.id)

        assert memory is not None
        assert memory.id == sample_memory.id

    @pytest.mark.asyncio
    async def test_get_memory_nonexistent(self, database, connection, mock_driver, mock_session):
        """Test retrieving a non-existent memory."""
        mock_session.execute_read = create_mock_execute([])
        _wire_session(mock_driver, mock_session)

        memory = await database.get_memory("nonexistent-id")
        assert memory is None

    @pytest.mark.asyncio
    async def test_search_memories_basic(self, database, connection, mock_driver, mock_session):
        """Test basic memory search."""
        now = datetime.now(timezone.utc).isoformat()
        search_results = [
            {
                "m": {
                    "id": str(uuid.uuid4()),
                    "type": "solution",
                    "title": "Test Solution",
                    "content": "Test content",
                    "tags": ["test"],
                    "importance": 0.8,
                    "confidence": 0.9,
                    "created_at": now,
                    "updated_at": now,
                },
            },
        ]
        mock_session.execute_read = create_mock_execute(search_results)
        _wire_session(mock_driver, mock_session)

        search_query = SearchQuery(query="test", memory_types=[MemoryType.SOLUTION])
        results = await database.search_memories(search_query)

        assert len(results) > 0
        assert results[0].type == MemoryType.SOLUTION

    @pytest.mark.asyncio
    async def test_search_memories_with_filters(self, database, connection, mock_driver, mock_session):
        """Test memory search with multiple filters."""
        mock_session.execute_read = create_mock_execute([])
        _wire_session(mock_driver, mock_session)

        search_query = SearchQuery(
            query="test",
            memory_types=[MemoryType.SOLUTION],
            tags=["python", "testing"],
            min_importance=0.7,
            limit=10,
        )
        results = await database.search_memories(search_query)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_update_memory(self, database, connection, sample_memory, mock_driver, mock_session):
        """Test updating a memory."""
        mock_session.execute_write = create_mock_execute([{"updated": 1}])
        _wire_session(mock_driver, mock_session)

        sample_memory.title = "Updated Title"
        success = await database.update_memory(sample_memory)
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_memory(self, database, connection, sample_memory, mock_driver, mock_session):
        """Test deleting a memory."""
        mock_session.execute_read = create_mock_execute([{"id": sample_memory.id}])
        mock_session.execute_write = create_mock_execute([])
        _wire_session(mock_driver, mock_session)

        success = await database.delete_memory(sample_memory.id)
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, database, connection, mock_driver, mock_session):
        """Test deleting a non-existent memory returns False."""
        mock_session.execute_read = create_mock_execute([])
        _wire_session(mock_driver, mock_session)

        success = await database.delete_memory("nonexistent-id")
        assert success is False

    @pytest.mark.asyncio
    async def test_create_relationship(self, database, connection, mock_driver, mock_session):
        """Test creating a relationship between memories."""
        rel_id = str(uuid.uuid4())
        mock_session.execute_write = create_mock_execute([{"id": rel_id}])
        _wire_session(mock_driver, mock_session)

        from_id = str(uuid.uuid4())
        to_id = str(uuid.uuid4())

        relationship_id = await database.create_relationship(
            from_memory_id=from_id,
            to_memory_id=to_id,
            relationship_type=RelationshipType.SOLVES,
            properties=RelationshipProperties(strength=0.9, confidence=0.8),
        )

        assert relationship_id == rel_id

    @pytest.mark.asyncio
    async def test_relationship_properties_validation(self, database):
        """Test that RelationshipProperties stores values correctly."""
        props = RelationshipProperties(strength=0.9, confidence=0.8)
        assert props.strength == 0.9
        assert props.confidence == 0.8

    @pytest.mark.asyncio
    async def test_get_related_memories(self, database, connection, mock_driver, mock_session):
        """Test getting related memories with depth traversal."""
        memory_id = str(uuid.uuid4())
        related_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        related_data = [
            {
                "related": {
                    "id": related_id,
                    "type": "problem",
                    "title": "Related Problem",
                    "content": "Related content",
                    "tags": [],
                    "importance": 0.7,
                    "confidence": 0.8,
                    "created_at": now,
                    "updated_at": now,
                },
                "rel_type": "SOLVES",
                "rel_props": {
                    "strength": 0.9,
                    "confidence": 0.8,
                    "evidence_count": 1,
                },
                "from_id": memory_id,
                "to_id": related_id,
            },
        ]
        mock_session.execute_read = create_mock_execute(related_data)
        _wire_session(mock_driver, mock_session)

        related = await database.get_related_memories(
            memory_id=memory_id,
            relationship_types=[RelationshipType.SOLVES],
            max_depth=2,
        )
        assert isinstance(related, list)

    @pytest.mark.asyncio
    async def test_get_related_memories_empty(self, database, connection, mock_driver, mock_session):
        """Test relationship traversal returns empty list when no related memories exist."""
        mock_session.execute_read = create_mock_execute([])
        _wire_session(mock_driver, mock_session)

        memory_id = str(uuid.uuid4())
        related = await database.get_related_memories(
            memory_id=memory_id,
            relationship_types=[],
            max_depth=1,
        )

        assert related == []

    @pytest.mark.asyncio
    async def test_get_memory_statistics(self, database, connection, mock_driver, mock_session):
        """Test getting database statistics."""
        mock_session.execute_read = create_mock_execute([{
            "total_memories": 100,
            "total_relationships": 250,
            "memory_types": {"solution": 50, "problem": 30, "task": 20},
            "relationship_types": {"SOLVES": 100, "RELATED_TO": 150},
        }])
        _wire_session(mock_driver, mock_session)

        stats = await database.get_memory_statistics()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, database, connection, sample_memory, mock_driver, mock_session):
        """Test concurrent database operations."""
        mock_session.execute_write = create_mock_execute([{"id": sample_memory.id}])
        _wire_session(mock_driver, mock_session)

        memories = [
            Memory(
                id=str(uuid.uuid4()),
                type=MemoryType.TASK,
                title=f"Task {i}",
                content=f"Task content {i}",
                tags=["test"],
            )
            for i in range(5)
        ]

        results = await asyncio.gather(
            *[database.store_memory(m) for m in memories]
        )
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test error handling and edge cases."""

    @neo4j_skip
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, database, connection, mock_driver, mock_session):
        """Test that failed transactions raise DatabaseConnectionError."""
        from neo4j.exceptions import Neo4jError

        mock_session.execute_write = AsyncMock(
            side_effect=Neo4jError("Transaction failed"),
        )
        _wire_session(mock_driver, mock_session)

        with pytest.raises(DatabaseConnectionError):
            await connection.execute_write_query("INVALID QUERY", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
