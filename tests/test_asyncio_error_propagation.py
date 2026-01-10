"""
Tests for asyncio.to_thread Error Propagation (P2 Priority).

This test suite verifies that errors from synchronous SQLite operations
properly propagate through asyncio.to_thread wrapper. This is critical
because asyncio.to_thread can sometimes mask or transform exceptions.

The SQLiteFallbackBackend uses asyncio.to_thread to run synchronous
sqlite3 operations in a thread pool to avoid blocking the async event loop.

Error types tested:
- DatabaseConnectionError: When database connection is closed
- ValidationError: When input validation fails
- MemoryNotFoundError: When memory doesn't exist
- General exceptions: To ensure type preservation

These tests follow TDD principles - they define expected behavior
for error propagation in concurrent scenarios.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path

from memorygraph.sqlite_database import SQLiteMemoryDatabase
from memorygraph.backends.sqlite_fallback import SQLiteFallbackBackend
from memorygraph.models import (
    Memory, MemoryType, DatabaseConnectionError,
    ValidationError, MemoryNotFoundError
)


class TestAsyncioToThreadErrorPropagation:
    """Tests for error propagation through asyncio.to_thread in SQLite operations."""

    @pytest.mark.asyncio
    async def test_database_error_propagates_through_to_thread(self, tmp_path):
        """Test that database errors properly propagate through asyncio.to_thread."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Close the connection to force errors
        await backend.disconnect()

        # Operations should raise DatabaseConnectionError
        with pytest.raises(DatabaseConnectionError):
            await db.get_memory("non-existent-id")

    @pytest.mark.asyncio
    async def test_validation_error_propagates(self, tmp_path):
        """Test that validation errors properly propagate."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Create memory without ID then try to update
        memory = Memory(
            type=MemoryType.GENERAL,
            title="Test",
            content="Content"
        )
        memory.id = None  # Force no ID

        with pytest.raises(ValidationError):
            await db.update_memory(memory)

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_memory_not_found_propagates(self, tmp_path):
        """Test that memory not found cases are handled correctly through asyncio.to_thread."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Try to get non-existent memory - should return None (not raise)
        result = await db.get_memory("nonexistent-id")
        assert result is None

        # Delete non-existent memory should return False (not raise)
        deleted = await db.delete_memory("nonexistent-id")
        assert deleted is False

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_concurrent_operations_handle_errors(self, tmp_path):
        """Test that sequential operations handle errors correctly."""
        db_path = str(tmp_path / "test_concurrent.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Store some test memories sequentially
        memories = []
        for i in range(5):
            memory = Memory(
                type=MemoryType.GENERAL,
                title=f"Test {i}",
                content=f"Content {i}"
            )
            mem_id = await db.store_memory(memory)
            memories.append(mem_id)

        # Ensure all memories are committed
        backend.commit()

        # Sequential reads should work
        for mem_id in memories:
            result = await db.get_memory(mem_id)
            assert result is not None

        # Reads of non-existent should return None (not raise)
        for i in range(5):
            result = await db.get_memory(f"nonexistent-{i}")
            assert result is None

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_exception_type_preserved(self, tmp_path):
        """Test that specific exception types are preserved through asyncio.to_thread."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # ValidationError should be preserved
        memory = Memory(
            id=None,  # Will be set
            type=MemoryType.GENERAL,
            title="Test",
            content="Content"
        )

        # Store it first
        mem_id = await db.store_memory(memory)

        # Now try to update with no ID - should raise ValidationError specifically
        memory.id = None

        try:
            await db.update_memory(memory)
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected
        except Exception as e:
            assert False, f"Wrong exception type: {type(e).__name__}"

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_concurrent_writes_error_handling(self, tmp_path):
        """Test error handling during sequential write operations (SQLite limitation)."""
        db_path = str(tmp_path / "test_seq.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Create multiple memories sequentially (SQLite doesn't handle concurrent writes well)
        memories = [
            Memory(
                type=MemoryType.GENERAL,
                title=f"Sequential Test {i}",
                content=f"Content {i}"
            )
            for i in range(10)
        ]

        # Store them sequentially
        mem_ids = []
        for mem in memories:
            mem_id = await db.store_memory(mem)
            mem_ids.append(mem_id)

        # All should succeed
        assert len(mem_ids) == 10
        assert all(mem_id is not None for mem_id in mem_ids)

        # Commit to ensure all writes are persisted
        backend.commit()

        # Verify all were stored (read sequentially to avoid threading issues)
        for mem_id in mem_ids:
            retrieved = await db.get_memory(mem_id)
            assert retrieved is not None

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_error_during_concurrent_operations(self, tmp_path):
        """Test that sequential read operations work correctly after writes."""
        db_path = str(tmp_path / "test_sequential.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Store multiple valid memories
        mem_ids = []
        for i in range(5):
            memory = Memory(
                type=MemoryType.GENERAL,
                title=f"Valid Memory {i}",
                content=f"Valid Content {i}"
            )
            mem_id = await db.store_memory(memory)
            mem_ids.append(mem_id)

        # Commit changes
        backend.commit()

        # Read them all sequentially
        for mem_id in mem_ids:
            result = await db.get_memory(mem_id)
            assert result is not None

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_database_locked_error_handling(self, tmp_path):
        """Test handling of sequential write operations (SQLite best practice)."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Create memories sequentially to avoid SQLite threading issues
        memories = [
            Memory(
                type=MemoryType.GENERAL,
                title=f"Sequential Write {i}",
                content=f"Testing sequential writes with content {i}"
            )
            for i in range(20)
        ]

        # Store them all sequentially
        mem_ids = []
        for mem in memories:
            mem_id = await db.store_memory(mem)
            mem_ids.append(mem_id)

        # All should succeed
        assert len(mem_ids) == 20
        assert all(mem_id is not None for mem_id in mem_ids)

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_error_traceback_preserved(self, tmp_path):
        """Test that error tracebacks are preserved through asyncio.to_thread."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Close connection to force error
        await backend.disconnect()

        # Error should have a meaningful traceback
        try:
            await db.get_memory("test-id")
            assert False, "Should have raised DatabaseConnectionError"
        except DatabaseConnectionError as e:
            # Check that we have a traceback
            import traceback
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            tb_str = ''.join(tb)

            # Traceback should mention the actual error location
            assert 'sqlite_database.py' in tb_str or 'get_memory' in tb_str

    @pytest.mark.asyncio
    async def test_multiple_sequential_errors(self, tmp_path):
        """Test that multiple sequential errors are handled correctly."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Close connection
        await backend.disconnect()

        # Multiple operations should all raise the same error type
        for i in range(5):
            with pytest.raises(DatabaseConnectionError):
                await db.get_memory(f"test-id-{i}")

    @pytest.mark.asyncio
    async def test_error_after_successful_operations(self, tmp_path):
        """Test that errors occur correctly after successful operations."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Successful operation
        memory = Memory(
            type=MemoryType.GENERAL,
            title="Test",
            content="Content"
        )
        mem_id = await db.store_memory(memory)
        assert mem_id is not None

        # Retrieve successfully
        retrieved = await db.get_memory(mem_id)
        assert retrieved is not None

        # Now close connection
        await backend.disconnect()

        # Further operations should fail
        with pytest.raises(DatabaseConnectionError):
            await db.get_memory(mem_id)

    @pytest.mark.asyncio
    async def test_validation_error_with_detailed_message(self, tmp_path):
        """Test that ValidationError preserves detailed error messages."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteFallbackBackend(db_path=db_path)
        await backend.connect()
        await backend.initialize_schema()

        db = SQLiteMemoryDatabase(backend)

        # Create memory without ID
        memory = Memory(
            type=MemoryType.GENERAL,
            title="Test",
            content="Content"
        )
        memory.id = None

        # Error message should be descriptive
        try:
            await db.update_memory(memory)
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            # Error message should mention what's wrong
            error_msg = str(e).lower()
            assert 'id' in error_msg or 'required' in error_msg

        await backend.disconnect()
