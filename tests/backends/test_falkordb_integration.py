"""
Integration tests for FalkorDB backend.

Requires a running FalkorDB instance:
    docker run -p 6379:6379 -it --rm falkordb/falkordb:latest
"""

import asyncio
import os

import pytest

from memorygraph.backends.falkordb_backend import FalkorDBBackend
from memorygraph.models import (
    Memory,
    MemoryType,
    RelationshipProperties,
    RelationshipType,
    SearchQuery,
)


def is_falkordb_available():
    try:
        from unittest.mock import MagicMock

        import falkordb

        if isinstance(falkordb, MagicMock) or isinstance(falkordb.FalkorDB, MagicMock):
            return False
        if not hasattr(falkordb, 'FalkorDB') or not callable(falkordb.FalkorDB):
            return False

        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", "6379"))
        try:
            client = falkordb.FalkorDB(host=host, port=port)
            graph = client.select_graph("test_connection")
            graph.query("RETURN 1")
            return True
        except Exception:
            return False
    except ImportError:
        return False


FALKORDB_AVAILABLE = is_falkordb_available()
skip_if_no_falkordb = pytest.mark.skipif(
    not FALKORDB_AVAILABLE,
    reason="FalkorDB not available. Run: docker run -p 6379:6379 -it --rm falkordb/falkordb:latest"
)


@pytest.fixture
async def falkordb_backend():
    if not FALKORDB_AVAILABLE:
        pytest.skip("FalkorDB not available")

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))

    backend = FalkorDBBackend(host=host, port=port, graph_name="test_memorygraph")
    await backend.connect()
    await backend.initialize_schema()

    yield backend

    try:
        await backend.execute_query("MATCH (n) DETACH DELETE n", write=True)
    except Exception:
        pass
    await backend.disconnect()


@skip_if_no_falkordb
class TestFalkorDBIntegration:

    @pytest.mark.asyncio
    async def test_full_memory_lifecycle(self, falkordb_backend):
        memory = Memory(
            type=MemoryType.SOLUTION,
            title="Redis Connection Pool Fix",
            content="Increased max connections from 10 to 50 to handle concurrent requests",
            tags=["redis", "performance", "connection-pool"],
            importance=0.9,
            confidence=0.95
        )

        memory_id = await falkordb_backend.store_memory(memory)
        assert memory_id is not None
        assert memory_id == memory.id

        retrieved = await falkordb_backend.get_memory(memory_id)
        assert retrieved is not None
        assert retrieved.id == memory_id
        assert retrieved.title == "Redis Connection Pool Fix"
        assert retrieved.type == MemoryType.SOLUTION
        assert "redis" in retrieved.tags

        retrieved.title = "Updated Redis Connection Pool Fix"
        retrieved.importance = 1.0
        assert await falkordb_backend.update_memory(retrieved) is True

        updated = await falkordb_backend.get_memory(memory_id)
        assert updated.title == "Updated Redis Connection Pool Fix"
        assert updated.importance == 1.0

        assert await falkordb_backend.delete_memory(memory_id) is True
        assert await falkordb_backend.get_memory(memory_id) is None

    @pytest.mark.asyncio
    async def test_relationship_creation_and_traversal(self, falkordb_backend):
        problem = Memory(
            type=MemoryType.PROBLEM,
            title="High API Latency",
            content="API endpoints showing 2-3 second response times",
            tags=["api", "performance"],
            importance=0.8
        )
        solution = Memory(
            type=MemoryType.SOLUTION,
            title="Implemented Caching Layer",
            content="Added Redis caching for frequently accessed data",
            tags=["redis", "caching", "performance"],
            importance=0.9
        )

        problem_id = await falkordb_backend.store_memory(problem)
        solution_id = await falkordb_backend.store_memory(solution)

        props = RelationshipProperties(
            strength=0.95,
            confidence=0.9,
            context="Caching reduced API latency from 2-3s to 50-100ms"
        )
        rel_id = await falkordb_backend.create_relationship(
            from_memory_id=solution_id,
            to_memory_id=problem_id,
            relationship_type=RelationshipType.SOLVES,
            properties=props
        )
        assert rel_id is not None

        related = await falkordb_backend.get_related_memories(problem_id)
        assert len(related) > 0

        found_solution = False
        for related_memory, relationship in related:
            if related_memory.id == solution_id:
                found_solution = True
                assert relationship.type == RelationshipType.SOLVES
                assert relationship.properties.strength == 0.95
                break
        assert found_solution, "Solution should be in related memories"

    @pytest.mark.asyncio
    async def test_search_functionality(self, falkordb_backend):
        memories_data = [
            ("Redis Timeout Issue", "Connection timeouts after 30 seconds", ["redis", "timeout"]),
            ("Database Query Optimization", "Optimized slow queries using indexes", ["database", "performance"]),
            ("Redis Cache Implementation", "Implemented Redis for session storage", ["redis", "caching"]),
        ]

        for title, content, tags in memories_data:
            memory = Memory(
                type=MemoryType.SOLUTION,
                title=title,
                content=content,
                tags=tags,
                importance=0.7
            )
            await falkordb_backend.store_memory(memory)

        results = await falkordb_backend.search_memories(SearchQuery(query="redis", limit=10))
        assert len(results) >= 2
        assert any("Redis" in m.title for m in results)

        results = await falkordb_backend.search_memories(SearchQuery(tags=["timeout"], limit=10))
        assert len(results) >= 1
        assert any("Timeout" in m.title for m in results)

        results = await falkordb_backend.search_memories(
            SearchQuery(memory_types=[MemoryType.SOLUTION], limit=10)
        )
        assert len(results) >= 3
        assert all(m.type == MemoryType.SOLUTION for m in results)

    @pytest.mark.asyncio
    async def test_statistics(self, falkordb_backend):
        for i in range(3):
            memory = Memory(
                type=MemoryType.SOLUTION if i % 2 == 0 else MemoryType.PROBLEM,
                title=f"Test Memory {i}",
                content=f"Content {i}",
                tags=["test"],
                importance=0.5
            )
            await falkordb_backend.store_memory(memory)

        stats = await falkordb_backend.get_memory_statistics()

        assert "total_memories" in stats
        assert stats["total_memories"]["count"] >= 3
        assert "memories_by_type" in stats
        assert len(stats["memories_by_type"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, falkordb_backend):
        async def create_memory(index):
            memory = Memory(
                type=MemoryType.SOLUTION,
                title=f"Concurrent Memory {index}",
                content=f"Content {index}",
                tags=["concurrent", "test"],
                importance=0.5
            )
            return await falkordb_backend.store_memory(memory)

        memory_ids = await asyncio.gather(*[create_memory(i) for i in range(10)])

        assert len(memory_ids) == 10
        assert all(mid is not None for mid in memory_ids)

        for mem_id in memory_ids:
            assert await falkordb_backend.get_memory(mem_id) is not None

    @pytest.mark.asyncio
    async def test_health_check(self, falkordb_backend):
        health = await falkordb_backend.health_check()

        assert health["connected"] is True
        assert health["backend_type"] == "falkordb"
        assert "host" in health
        assert "port" in health
        assert "statistics" in health
