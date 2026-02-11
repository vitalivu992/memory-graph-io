"""Shared fixtures and helpers for backend tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest


def make_falkordb_node(properties: dict) -> Mock:
    node = Mock()
    node.properties = properties
    return node


def make_falkordb_result(header_names: list, rows: list) -> Mock:
    """Create a mock FalkorDB QueryResult matching the real header format."""
    result = Mock()
    # FalkorDB header format: [[ColumnType, column_name], ...]
    # ColumnType is an int constant; we use 1 as a placeholder
    result.header = [[1, name] for name in header_names]
    result.result_set = rows
    return result


def make_memory_node(
    id: str,
    *,
    type: str = "solution",
    title: str = "Test",
    content: str = "Content",
    tags: list | None = None,
    importance: float = 0.8,
    confidence: float = 0.9,
    summary: str | None = None,
) -> Mock:
    """Create a mock FalkorDB node with memory properties."""
    now = datetime.now(timezone.utc).isoformat()
    return make_falkordb_node({
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


def make_connected_backend(backend_cls, **kwargs):
    """Create a pre-connected backend with mocked client and graph.

    For FalkorDBBackend, pass host/port kwargs.
    For FalkorDBLiteBackend, pass db_path kwarg.
    """
    backend = backend_cls(**kwargs)
    backend.client = Mock()
    backend.graph = Mock()
    backend._connected = True
    return backend


@pytest.fixture
def mock_memgraph_driver():
    mock_driver = AsyncMock()
    mock_driver.verify_connectivity = AsyncMock()
    mock_driver.close = AsyncMock()
    return mock_driver


@pytest.fixture
def mock_memgraph_session():
    mock_session = AsyncMock()
    mock_session.close = AsyncMock()
    return mock_session


@pytest.fixture
def mock_memgraph_transaction():
    mock_tx = AsyncMock()
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[])
    mock_tx.run = AsyncMock(return_value=mock_result)
    return mock_tx


@pytest.fixture
def mock_memgraph_database(mock_memgraph_driver, mock_memgraph_session, mock_memgraph_transaction):
    """Returns (mock_db_class, mock_driver, mock_session, mock_tx) for patching AsyncGraphDatabase."""

    async def execute_write_side_effect(fn, *args):
        return await fn(mock_memgraph_transaction, *args)

    mock_memgraph_session.execute_write = AsyncMock(side_effect=execute_write_side_effect)
    mock_memgraph_driver.session = Mock(return_value=mock_memgraph_session)

    mock_db_class = Mock()
    mock_db_class.driver.return_value = mock_memgraph_driver

    return mock_db_class, mock_memgraph_driver, mock_memgraph_session, mock_memgraph_transaction
