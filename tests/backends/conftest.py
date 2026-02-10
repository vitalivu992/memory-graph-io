"""
Shared fixtures for backend tests.

This file contains reusable fixtures for mocking backends,
especially for Memgraph and Neo4j-based backends,
and shared helpers for FalkorDB mock result construction.
"""

import pytest
from unittest.mock import AsyncMock, Mock


# ---------------------------------------------------------------------------
# FalkorDB shared mock helpers
# ---------------------------------------------------------------------------

def make_falkordb_node(properties: dict) -> Mock:
    """Create a mock FalkorDB Node with a properties dict."""
    node = Mock()
    node.properties = properties
    return node


def make_falkordb_result(header_names: list, rows: list) -> Mock:
    """
    Create a mock FalkorDB QueryResult matching the real format.

    Args:
        header_names: List of column name strings (e.g., ["id", "m"])
        rows: List of lists, each inner list is a row of values
    """
    result = Mock()
    # FalkorDB header format: [[ColumnType, column_name], ...]
    # ColumnType is an int constant; we use 1 as a placeholder
    result.header = [[1, name] for name in header_names]
    result.result_set = rows
    return result


@pytest.fixture
def mock_memgraph_driver():
    """Create a mock Memgraph/Neo4j driver with common setup."""
    mock_driver = AsyncMock()
    mock_driver.verify_connectivity = AsyncMock()
    mock_driver.close = AsyncMock()
    return mock_driver


@pytest.fixture
def mock_memgraph_session():
    """Create a mock Memgraph/Neo4j session with common setup."""
    mock_session = AsyncMock()
    mock_session.close = AsyncMock()
    return mock_session


@pytest.fixture
def mock_memgraph_transaction():
    """Create a mock Memgraph/Neo4j transaction with common setup."""
    mock_tx = AsyncMock()
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[])
    mock_tx.run = AsyncMock(return_value=mock_result)
    return mock_tx


@pytest.fixture
def mock_memgraph_database(mock_memgraph_driver, mock_memgraph_session, mock_memgraph_transaction):
    """
    Create a complete mock Memgraph/Neo4j database setup.

    Returns a tuple of (mock_db_class, mock_driver, mock_session, mock_tx)
    suitable for patching AsyncGraphDatabase.
    """

    async def execute_write_side_effect(fn, *args):
        """Execute the transaction function with the mock transaction."""
        return await fn(mock_memgraph_transaction, *args)

    mock_memgraph_session.execute_write = AsyncMock(side_effect=execute_write_side_effect)
    mock_memgraph_driver.session = Mock(return_value=mock_memgraph_session)

    mock_db_class = Mock()
    mock_db_class.driver.return_value = mock_memgraph_driver

    return (mock_db_class, mock_memgraph_driver, mock_memgraph_session, mock_memgraph_transaction)
