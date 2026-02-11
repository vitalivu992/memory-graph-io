"""
Integration tests for FalkorDBLite backend.

Tests are skipped if falkordblite is not installed or if the bundled binaries
are incompatible with the current platform (e.g., Linux binaries on macOS).
"""

import os
import platform
import subprocess
import tempfile

import pytest

FALKORDBLITE_AVAILABLE = False
SKIP_REASON = "falkordblite not installed"

try:
    from unittest.mock import MagicMock

    from redislite.client import __falkordb_module__, __redis_executable__
    from redislite.falkordb_client import FalkorDB

    if isinstance(FalkorDB, MagicMock) or not callable(FalkorDB):
        SKIP_REASON = "falkordblite is mocked"
    elif not __redis_executable__ or not os.path.exists(__redis_executable__):
        SKIP_REASON = "redis-server executable not found"
    elif not __falkordb_module__ or not os.path.exists(__falkordb_module__):
        SKIP_REASON = "falkordb.so module not found"
    else:
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["file", __falkordb_module__],
                    capture_output=True, text=True, timeout=5
                )
                if "ELF" in result.stdout:
                    SKIP_REASON = (
                        "falkordb.so is a Linux ELF binary, incompatible with macOS - "
                        "falkordblite v0.4.0+ only supports Linux"
                    )
                elif "Mach-O" in result.stdout:
                    FALKORDBLITE_AVAILABLE = True
                else:
                    SKIP_REASON = f"unknown falkordb.so format: {result.stdout.strip()}"
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                SKIP_REASON = f"failed to check falkordb.so compatibility: {e}"
        else:
            FALKORDBLITE_AVAILABLE = True
except ImportError as e:
    SKIP_REASON = f"falkordblite import failed: {e}"

from memorygraph.backends.falkordblite_backend import FalkorDBLiteBackend
from memorygraph.models import (
    Memory,
    MemoryContext,
    MemoryType,
    RelationshipProperties,
    RelationshipType,
    SearchQuery,
)


@pytest.mark.skipif(not FALKORDBLITE_AVAILABLE, reason=SKIP_REASON)
class TestFalkorDBLiteIntegration:

    @pytest.fixture
    async def backend(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            backend = FalkorDBLiteBackend(db_path=db_path)
            await backend.connect()
            await backend.initialize_schema()
            yield backend
            await backend.disconnect()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_store_and_retrieve_memory(self, backend):
        memory = Memory(
            type=MemoryType.SOLUTION,
            title="Test Memory",
            content="This is a test memory for FalkorDBLite",
            tags=["test", "integration"],
            importance=0.8,
            confidence=0.9
        )

        memory_id = await backend.store_memory(memory)
        assert memory_id is not None

        retrieved = await backend.get_memory(memory_id)
        assert retrieved is not None
        assert retrieved.title == "Test Memory"
        assert retrieved.content == "This is a test memory for FalkorDBLite"
        assert "test" in retrieved.tags
        assert "integration" in retrieved.tags

    @pytest.mark.asyncio
    async def test_update_memory(self, backend):
        memory = Memory(
            type=MemoryType.PROBLEM,
            title="Original Title",
            content="Original content",
            tags=["original"],
            importance=0.5,
            confidence=0.7
        )
        memory_id = await backend.store_memory(memory)

        memory.title = "Updated Title"
        memory.content = "Updated content"
        memory.tags = ["updated"]
        memory.importance = 0.9

        assert await backend.update_memory(memory)

        retrieved = await backend.get_memory(memory_id)
        assert retrieved.title == "Updated Title"
        assert retrieved.content == "Updated content"
        assert "updated" in retrieved.tags
        assert retrieved.importance == 0.9

    @pytest.mark.asyncio
    async def test_delete_memory(self, backend):
        memory = Memory(
            type=MemoryType.SOLUTION,
            title="To Be Deleted",
            content="This memory will be deleted",
            tags=["delete"],
            importance=0.5,
            confidence=0.5
        )
        memory_id = await backend.store_memory(memory)

        assert await backend.get_memory(memory_id) is not None
        assert await backend.delete_memory(memory_id)
        assert await backend.get_memory(memory_id) is None

    @pytest.mark.asyncio
    async def test_search_memories(self, backend):
        memories_data = [
            (MemoryType.SOLUTION, "Python Async Solution", "How to use async/await in Python", ["python", "async"], 0.8),
            (MemoryType.PROBLEM, "JavaScript Async Problem", "Promise chain is too complex", ["javascript", "async"], 0.6),
            (MemoryType.SOLUTION, "Database Query Optimization", "Added index to improve query performance", ["database", "performance"], 0.7),
        ]
        for mem_type, title, content, tags, importance in memories_data:
            await backend.store_memory(Memory(
                type=mem_type, title=title, content=content,
                tags=tags, importance=importance, confidence=0.8
            ))

        results = await backend.search_memories(SearchQuery(query="async"))
        assert len(results) >= 2

        results = await backend.search_memories(SearchQuery(tags=["python"]))
        assert len(results) >= 1
        assert any("python" in r.tags for r in results)

        results = await backend.search_memories(SearchQuery(memory_types=[MemoryType.SOLUTION]))
        assert len(results) >= 2
        assert all(r.type == MemoryType.SOLUTION for r in results)

        results = await backend.search_memories(SearchQuery(min_importance=0.75))
        assert len(results) >= 1
        assert all(r.importance >= 0.75 for r in results)

    @pytest.mark.asyncio
    async def test_create_and_retrieve_relationships(self, backend):
        problem = Memory(
            type=MemoryType.PROBLEM,
            title="Connection Timeout",
            content="Database connections timing out",
            tags=["database", "timeout"],
            importance=0.8, confidence=0.9
        )
        problem_id = await backend.store_memory(problem)

        solution = Memory(
            type=MemoryType.SOLUTION,
            title="Increase Connection Pool",
            content="Increased max connections to 50",
            tags=["database", "fix"],
            importance=0.9, confidence=0.95
        )
        solution_id = await backend.store_memory(solution)

        props = RelationshipProperties(
            strength=0.9, confidence=0.85,
            context="This solution fixed the timeout problem"
        )
        rel_id = await backend.create_relationship(
            from_memory_id=solution_id,
            to_memory_id=problem_id,
            relationship_type=RelationshipType.SOLVES,
            properties=props
        )
        assert rel_id is not None

        related = await backend.get_related_memories(problem_id)
        assert len(related) >= 1

        found = any(
            memory.id == solution_id and relationship.type == RelationshipType.SOLVES
            for memory, relationship in related
        )
        assert found, "Related memory not found"

    @pytest.mark.asyncio
    async def test_memory_with_context(self, backend):
        context = MemoryContext(
            project_path="/test/project",
            file_path="/test/project/main.py",
            function_name="process_data",
            line_numbers=[42, 43, 44],
            user="test_user",
            session_id="test_session_123",
            additional_metadata={"version": "1.0.0", "environment": "test"}
        )

        memory = Memory(
            type=MemoryType.CODE_PATTERN,
            title="Data Processing Pattern",
            content="Efficient data processing using generators",
            tags=["python", "generators", "performance"],
            importance=0.85, confidence=0.9,
            context=context
        )

        memory_id = await backend.store_memory(memory)
        retrieved = await backend.get_memory(memory_id)

        assert retrieved is not None
        assert retrieved.context is not None
        assert retrieved.context.project_path == "/test/project"
        assert retrieved.context.file_path == "/test/project/main.py"
        assert retrieved.context.function_name == "process_data"
        assert retrieved.context.line_numbers == [42, 43, 44]
        assert retrieved.context.user == "test_user"
        assert retrieved.context.additional_metadata.get("version") == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_memory_statistics(self, backend):
        for i in range(5):
            await backend.store_memory(Memory(
                type=MemoryType.SOLUTION if i % 2 == 0 else MemoryType.PROBLEM,
                title=f"Memory {i}",
                content=f"Content {i}",
                tags=["test"],
                importance=0.5 + (i * 0.1),
                confidence=0.8
            ))

        stats = await backend.get_memory_statistics()

        assert stats["total_memories"]["count"] >= 5
        assert len(stats["memories_by_type"]) >= 1

    @pytest.mark.asyncio
    async def test_health_check(self, backend):
        health = await backend.health_check()

        assert health["connected"] is True
        assert health["backend_type"] == "falkordblite"
        assert "db_path" in health
        assert "graph_name" in health
        assert "statistics" in health

    @pytest.mark.asyncio
    async def test_backend_capabilities(self, backend):
        assert backend.backend_name() == "falkordblite"
        assert backend.supports_fulltext_search() is True
        assert backend.supports_transactions() is True

    @pytest.mark.asyncio
    async def test_multiple_relationships(self, backend):
        mem1 = Memory(type=MemoryType.PROBLEM, title="Issue A", content="Description A",
                      tags=["issue"], importance=0.7, confidence=0.8)
        id1 = await backend.store_memory(mem1)

        mem2 = Memory(type=MemoryType.SOLUTION, title="Solution B", content="Description B",
                      tags=["solution"], importance=0.8, confidence=0.9)
        id2 = await backend.store_memory(mem2)

        mem3 = Memory(type=MemoryType.SOLUTION, title="Alternative C", content="Description C",
                      tags=["solution"], importance=0.75, confidence=0.85)
        id3 = await backend.store_memory(mem3)

        await backend.create_relationship(
            id2, id1, RelationshipType.SOLVES,
            RelationshipProperties(strength=0.9, confidence=0.9)
        )
        await backend.create_relationship(
            id3, id1, RelationshipType.SOLVES,
            RelationshipProperties(strength=0.8, confidence=0.85)
        )
        await backend.create_relationship(
            id3, id2, RelationshipType.ALTERNATIVE_TO,
            RelationshipProperties(strength=0.7, confidence=0.8)
        )

        related = await backend.get_related_memories(id1)
        assert len(related) >= 2

        rel_types = [rel.type for _, rel in related]
        assert RelationshipType.SOLVES in rel_types
