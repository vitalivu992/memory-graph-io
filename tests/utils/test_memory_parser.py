"""
Tests for memory_parser utility module.

This module provides comprehensive test coverage for the memory parsing
utilities used by multiple backends (Neo4j, FalkorDB, SQLite, etc.).
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import patch

from src.memorygraph.utils.memory_parser import (
    parse_memory_from_properties,
    _parse_datetime,
    _extract_context,
)
from src.memorygraph.models import Memory, MemoryType, MemoryContext


class TestParseMemoryFromProperties:
    """Tests for parse_memory_from_properties function."""

    def test_valid_memory_parses_correctly(self):
        """Test that valid node data converts to Memory object."""
        node_data = {
            "id": "test-123",
            "type": "solution",
            "title": "Test Solution",
            "content": "Test content",
            "summary": "Test summary",
            "tags": ["python", "testing"],
            "importance": 0.8,
            "confidence": 0.9,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-02T00:00:00+00:00",
        }

        result = parse_memory_from_properties(node_data, "test")

        assert result is not None
        assert result.id == "test-123"
        assert result.type == MemoryType.SOLUTION
        assert result.title == "Test Solution"
        assert result.content == "Test content"
        assert result.summary == "Test summary"
        assert result.tags == ["python", "testing"]
        assert result.importance == 0.8
        assert result.confidence == 0.9

    def test_missing_required_field_returns_none(self):
        """Test that missing required fields returns None."""
        node_data = {
            "id": "test-123",
            # missing "type"
            "title": "Test",
            "content": "Content",
        }

        result = parse_memory_from_properties(node_data, "test")
        assert result is None

    def test_invalid_memory_type_returns_none(self):
        """Test that invalid MemoryType value returns None."""
        node_data = {
            "id": "test-123",
            "type": "invalid_type_xyz",
            "title": "Test",
            "content": "Content",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }

        result = parse_memory_from_properties(node_data, "test")
        assert result is None

    def test_context_fields_extracted(self):
        """Test that context_ prefixed fields are extracted."""
        node_data = {
            "id": "test-123",
            "type": "solution",
            "title": "Test",
            "content": "Content",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "context_project_path": "/path/to/project",
            "context_languages": '["python", "javascript"]',
        }

        result = parse_memory_from_properties(node_data, "test")

        assert result is not None
        assert result.context is not None
        assert result.context.project_path == "/path/to/project"
        assert result.context.languages == ["python", "javascript"]

    def test_last_accessed_optional_field(self):
        """Test that optional last_accessed field is handled."""
        node_data = {
            "id": "test-123",
            "type": "solution",
            "title": "Test",
            "content": "Content",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "last_accessed": "2024-01-03T00:00:00+00:00",
        }

        result = parse_memory_from_properties(node_data, "test")

        assert result is not None
        assert result.last_accessed is not None
        assert result.last_accessed.day == 3

    def test_default_values_applied(self):
        """Test that default values are applied for missing optional fields."""
        node_data = {
            "id": "test-123",
            "type": "solution",
            "title": "Test",
            "content": "Content",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            # No importance, confidence, tags
        }

        result = parse_memory_from_properties(node_data, "test")

        assert result is not None
        assert result.importance == 0.5  # default
        assert result.confidence == 0.8  # default
        assert result.tags == []  # default

    def test_exception_logged_and_returns_none(self):
        """Test that exceptions are logged and None is returned."""
        node_data = {
            "id": "test-123",
            "type": "solution",
            "title": "Test",
            "content": "Content",
            "created_at": "not-a-valid-datetime",  # Will cause parsing error
            "updated_at": "2024-01-01T00:00:00+00:00",
        }

        with patch('src.memorygraph.utils.memory_parser.logger') as mock_logger:
            result = parse_memory_from_properties(node_data, "test_source")

            assert result is None
            mock_logger.error.assert_called_once()
            # Verify source name appears in error message
            error_call = str(mock_logger.error.call_args)
            assert "test_source" in error_call

    def test_usage_count_field(self):
        """Test that usage_count field is parsed."""
        node_data = {
            "id": "test-123",
            "type": "solution",
            "title": "Test",
            "content": "Content",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "usage_count": 42,
        }

        result = parse_memory_from_properties(node_data, "test")

        assert result is not None
        assert result.usage_count == 42

    def test_effectiveness_field(self):
        """Test that effectiveness field is parsed."""
        node_data = {
            "id": "test-123",
            "type": "solution",
            "title": "Test",
            "content": "Content",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "effectiveness": 0.95,
        }

        result = parse_memory_from_properties(node_data, "test")

        assert result is not None
        assert result.effectiveness == 0.95


class TestParseDatetime:
    """Tests for _parse_datetime helper function."""

    def test_iso_string_parsed(self):
        """Test ISO format string is parsed to datetime."""
        result = _parse_datetime("2024-01-15T10:30:00+00:00")

        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_datetime_passthrough(self):
        """Test datetime object is returned as-is."""
        dt = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        result = _parse_datetime(dt)

        assert result is dt

    def test_none_value_returns_none(self):
        """Test None value returns None."""
        result = _parse_datetime(None)
        assert result is None

    def test_iso_string_without_timezone(self):
        """Test ISO format string without timezone."""
        result = _parse_datetime("2024-01-15T10:30:00")

        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15


class TestExtractContext:
    """Tests for _extract_context helper function."""

    def test_extracts_context_prefix_fields(self):
        """Test fields with context_ prefix are extracted."""
        node_data = {
            "id": "test",
            "context_project_path": "/path",
            "context_git_branch": "main",
            "other_field": "ignored",
        }

        result = _extract_context(node_data)

        assert "project_path" in result
        assert result["project_path"] == "/path"
        assert "git_branch" in result
        assert result["git_branch"] == "main"
        assert "other_field" not in result

    def test_json_array_deserialized(self):
        """Test JSON array strings are deserialized."""
        node_data = {
            "context_languages": '["python", "rust"]',
        }

        result = _extract_context(node_data)

        assert result["languages"] == ["python", "rust"]

    def test_json_object_deserialized(self):
        """Test JSON object strings are deserialized."""
        node_data = {
            "context_additional_metadata": '{"key": "value"}',
        }

        result = _extract_context(node_data)

        assert result["additional_metadata"] == {"key": "value"}

    def test_invalid_json_kept_as_string(self):
        """Test invalid JSON is kept as string."""
        node_data = {
            "context_data": '{not valid json}',
        }

        result = _extract_context(node_data)

        assert result["data"] == '{not valid json}'

    def test_timestamp_string_converted(self):
        """Test timestamp field is converted to datetime."""
        node_data = {
            "context_timestamp": "2024-01-15T10:30:00+00:00",
        }

        result = _extract_context(node_data)

        assert isinstance(result["timestamp"], datetime)
        assert result["timestamp"].year == 2024
        assert result["timestamp"].month == 1
        assert result["timestamp"].day == 15

    def test_none_values_skipped(self):
        """Test None values are not included."""
        node_data = {
            "context_present": "value",
            "context_absent": None,
        }

        result = _extract_context(node_data)

        assert "present" in result
        assert "absent" not in result

    def test_json_array_without_additional_metadata(self):
        """Test JSON arrays in context fields other than additional_metadata."""
        node_data = {
            "context_files_involved": '["file1.py", "file2.py"]',
            "context_frameworks": '["fastapi", "pytest"]',
        }

        result = _extract_context(node_data)

        assert result["files_involved"] == ["file1.py", "file2.py"]
        assert result["frameworks"] == ["fastapi", "pytest"]

    def test_empty_context(self):
        """Test extraction with no context fields."""
        node_data = {
            "id": "test",
            "title": "Test",
            "content": "Content",
        }

        result = _extract_context(node_data)

        assert result == {}

    def test_context_with_mixed_types(self):
        """Test context extraction with mixed data types."""
        node_data = {
            "context_project_path": "/path/to/project",
            "context_languages": '["python", "javascript"]',
            "context_git_commit": "abc123",
            "context_additional_metadata": '{"build": "123", "env": "test"}',
        }

        result = _extract_context(node_data)

        assert result["project_path"] == "/path/to/project"
        assert result["languages"] == ["python", "javascript"]
        assert result["git_commit"] == "abc123"
        assert result["additional_metadata"] == {"build": "123", "env": "test"}
