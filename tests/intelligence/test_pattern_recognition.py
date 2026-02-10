"""Tests for pattern recognition functionality."""

from datetime import datetime, timezone

import pytest

from memorygraph.intelligence.pattern_recognition import (
    Pattern,
    PatternRecognizer,
    extract_patterns,
    find_similar_problems,
    suggest_patterns,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

class MockBackend:
    """Mock backend for testing pattern recognition."""

    def __init__(self):
        self.queries: list[tuple[str, dict]] = []
        self.test_data: dict[str, list] = {
            "similar_problems": [],
            "patterns": [],
            "suggestions": [],
            "co_occurrences": [],
        }

    async def execute_query(self, query: str, params: dict):
        self.queries.append((query, params))

        if "type: 'problem'" in query:
            return self.test_data["similar_problems"]
        if "WHERE id(e1) < id(e2)" in query:
            return self.test_data["co_occurrences"]
        if "MENTIONS" in query and "collect(m.id)" in query:
            return self.test_data["patterns"]
        if "UNWIND $entities" in query:
            return self.test_data["suggestions"]
        return []


class ErrorBackend:
    """Backend that always raises on execute_query."""

    async def execute_query(self, query: str, params: dict):
        raise Exception("Database error")


def _make_problem(
    problem_id: str = "p1",
    title: str = "Auth timeout",
    content: str = "Authentication times out after 30min",
    similarity: float = 0.85,
    solutions: list | None = None,
) -> dict:
    """Build a mock similar-problem result dict."""
    return {
        "problem_id": problem_id,
        "problem_title": title,
        "problem_content": content,
        "created_at": datetime.now(timezone.utc),
        "similarity": similarity,
        "solutions": solutions or [],
    }


def _make_entity_pattern(
    entity: str,
    entity_type: str = "technology",
    memory_ids: list[str] | None = None,
    occurrence_count: int = 3,
) -> dict:
    """Build a mock entity-pattern result dict."""
    return {
        "entity": entity,
        "entity_type": entity_type,
        "memory_ids": memory_ids or [f"m{i}" for i in range(1, occurrence_count + 1)],
        "occurrence_count": occurrence_count,
    }


def _make_suggestion(
    memory_id: str = "m1",
    memory_type: str = "solution",
    title: str = "React Authentication",
    content: str = "How to implement auth in React with hooks",
    matched_entities: list[str] | None = None,
    all_entity_texts: list[str] | None = None,
    match_count: int = 2,
) -> dict:
    """Build a mock suggestion result dict."""
    return {
        "memory_id": memory_id,
        "memory_type": memory_type,
        "title": title,
        "content": content,
        "matched_entities": matched_entities or ["React", "authentication"],
        "all_entity_texts": all_entity_texts or ["React", "authentication", "hooks"],
        "match_count": match_count,
    }


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
def recognizer(backend: MockBackend) -> PatternRecognizer:
    return PatternRecognizer(backend)


# ---------------------------------------------------------------------------
# Pattern model
# ---------------------------------------------------------------------------

class TestPatternModel:
    """Test Pattern model."""

    def test_pattern_creation(self):
        pattern = Pattern(
            id="pattern-1",
            name="Test Pattern",
            description="A test pattern",
            pattern_type="solution",
            confidence=0.8,
            occurrences=5,
        )
        assert pattern.id == "pattern-1"
        assert pattern.confidence == 0.8
        assert pattern.occurrences == 5

    def test_pattern_with_entities(self):
        pattern = Pattern(
            id="pattern-2",
            name="Auth Pattern",
            description="Authentication pattern",
            pattern_type="solution",
            confidence=0.9,
            entities=["Python", "JWT", "authentication"],
        )
        assert len(pattern.entities) == 3
        assert "JWT" in pattern.entities

    def test_pattern_confidence_validation(self):
        with pytest.raises(ValueError):
            Pattern(
                id="p",
                name="test",
                description="test",
                pattern_type="test",
                confidence=1.5,
            )


# ---------------------------------------------------------------------------
# PatternRecognizer
# ---------------------------------------------------------------------------

class TestPatternRecognizer:
    """Test PatternRecognizer class."""

    async def test_recognizer_initialization(self, backend, recognizer):
        assert recognizer.backend is backend

    async def test_find_similar_problems_empty(self, backend, recognizer):
        results = await recognizer.find_similar_problems(
            "Authentication error in API", threshold=0.7
        )
        assert results == []
        assert len(backend.queries) > 0

    async def test_find_similar_problems_with_results(self, backend, recognizer):
        backend.test_data["similar_problems"] = [
            _make_problem(
                solutions=[{
                    "id": "s1",
                    "title": "Increase timeout",
                    "content": "Set timeout to 1 hour",
                    "effectiveness": 0.9,
                }],
            )
        ]

        results = await recognizer.find_similar_problems("Auth timeout issue")

        assert len(results) == 1
        assert results[0]["problem_id"] == "p1"
        assert results[0]["similarity"] == 0.85

    async def test_extract_patterns_empty(self, backend, recognizer):
        patterns = await recognizer.extract_patterns("solution", min_occurrences=3)
        assert patterns == []
        assert len(backend.queries) > 0

    async def test_extract_patterns_with_entities(self, backend, recognizer):
        backend.test_data["patterns"] = [
            _make_entity_pattern("Python", memory_ids=["m1", "m2", "m3"]),
            _make_entity_pattern("FastAPI", memory_ids=["m1", "m2", "m4"]),
        ]

        patterns = await recognizer.extract_patterns("solution", min_occurrences=3)

        assert len(patterns) > 0
        pattern_names = {p.name for p in patterns}
        assert any("Python" in name for name in pattern_names)

    async def test_suggest_patterns_empty_context(self, recognizer):
        patterns = await recognizer.suggest_patterns("")
        assert patterns == []

    async def test_suggest_patterns_with_context(self, backend, recognizer):
        backend.test_data["suggestions"] = [_make_suggestion()]

        patterns = await recognizer.suggest_patterns(
            "Need to implement authentication in React application"
        )
        assert isinstance(patterns, list)

    def test_extract_keywords(self, recognizer):
        keywords = recognizer._extract_keywords(
            "The authentication system has a timeout error"
        )

        assert "authentication" in keywords
        assert "system" in keywords or "timeout" in keywords or "error" in keywords
        assert "the" not in keywords
        assert "has" not in keywords


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    async def test_find_similar_problems_function(self, backend):
        results = await find_similar_problems(
            backend, "API authentication error", threshold=0.6, limit=5
        )
        assert results == []
        assert len(backend.queries) > 0

    async def test_extract_patterns_function(self, backend):
        patterns = await extract_patterns(backend, "solution", min_occurrences=3)
        assert patterns == []

    async def test_suggest_patterns_function(self, backend):
        patterns = await suggest_patterns(backend, "Using Python with FastAPI")
        assert isinstance(patterns, list)


# ---------------------------------------------------------------------------
# Real-world scenarios
# ---------------------------------------------------------------------------

class TestRealWorldScenarios:
    """Test pattern recognition with real-world scenarios."""

    async def test_bug_pattern_recognition(self, backend, recognizer):
        backend.test_data["patterns"] = [
            _make_entity_pattern(
                "NullPointerException",
                entity_type="error",
                memory_ids=["b1", "b2", "b3", "b4"],
                occurrence_count=4,
            )
        ]

        patterns = await recognizer.extract_patterns("problem", min_occurrences=3)

        assert len(patterns) > 0
        assert patterns[0].occurrences >= 3

    async def test_solution_pattern_suggestion(self, backend, recognizer):
        backend.test_data["suggestions"] = [
            _make_suggestion(
                memory_id="s1",
                title="Caching Strategy",
                content="Use Redis for session caching",
                matched_entities=["Redis", "caching"],
                all_entity_texts=["Redis", "caching", "session"],
            )
        ]

        patterns = await recognizer.suggest_patterns(
            "Need to implement caching with Redis"
        )
        assert isinstance(patterns, list)

    async def test_technology_stack_patterns(self, backend, recognizer):
        backend.test_data["patterns"] = [
            _make_entity_pattern("React", memory_ids=["t1", "t2", "t3"]),
            _make_entity_pattern("TypeScript", memory_ids=["t1", "t2", "t3"]),
        ]

        patterns = await recognizer.extract_patterns("decision", min_occurrences=3)
        assert isinstance(patterns, list)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    async def test_empty_problem_text(self, recognizer):
        results = await recognizer.find_similar_problems("")
        assert results == []

    async def test_very_long_text(self, recognizer):
        keywords = recognizer._extract_keywords("authentication " * 1000)
        assert "authentication" in keywords

    async def test_special_characters(self, recognizer):
        keywords = recognizer._extract_keywords(
            "Error in @user/package-name with C++ code"
        )
        assert isinstance(keywords, list)
        assert "error" in keywords or "code" in keywords

    async def test_backend_error_handling(self):
        """All recognizer methods return [] on backend errors."""
        recognizer = PatternRecognizer(ErrorBackend())

        assert await recognizer.find_similar_problems("test problem") == []
        assert await recognizer.extract_patterns("solution") == []
        assert await recognizer.suggest_patterns("test context") == []


# ---------------------------------------------------------------------------
# Pattern quality & relevance
# ---------------------------------------------------------------------------

class TestPatternQuality:
    """Test pattern quality and relevance."""

    async def test_pattern_confidence_scoring(self, backend, recognizer):
        backend.test_data["patterns"] = [
            _make_entity_pattern(
                "Docker",
                memory_ids=["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"],
                occurrence_count=8,
            ),
            _make_entity_pattern(
                "Kubernetes",
                memory_ids=["m1", "m2"],
                occurrence_count=2,
            ),
        ]

        patterns = await recognizer.extract_patterns("solution", min_occurrences=2)

        if len(patterns) >= 2:
            docker_pattern = next((p for p in patterns if "Docker" in p.name), None)
            k8s_pattern = next(
                (p for p in patterns if "Kubernetes" in p.name), None
            )
            if docker_pattern and k8s_pattern:
                assert docker_pattern.confidence > k8s_pattern.confidence

    async def test_similarity_threshold_filtering(self, backend, recognizer):
        backend.test_data["similar_problems"] = [
            _make_problem(problem_id="p1", title="High similarity", similarity=0.9),
            _make_problem(problem_id="p2", title="Low similarity", similarity=0.5),
        ]

        results = await recognizer.find_similar_problems("test", threshold=0.8)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Error handling: _find_entity_co_occurrences
# ---------------------------------------------------------------------------

class TestPatternRecognitionErrorHandling:
    """Test error handling in _find_entity_co_occurrences."""

    async def test_find_entity_co_occurrences_handles_backend_error(self):
        backend = MockBackend()
        backend.execute_query = ErrorBackend().execute_query  # type: ignore[assignment]

        recognizer = PatternRecognizer(backend)
        result = await recognizer._find_entity_co_occurrences(
            "technology", min_occurrences=2
        )
        assert result == []

    async def test_find_entity_co_occurrences_with_results(self, backend):
        backend.test_data["co_occurrences"] = [
            {
                "entity1": "Python",
                "entity2": "pytest",
                "occurrence_count": 5,
                "memory_ids": ["m1", "m2", "m3"],
            },
            {
                "entity1": "FastAPI",
                "entity2": "async",
                "occurrence_count": 3,
                "memory_ids": ["m4", "m5"],
            },
        ]

        recognizer = PatternRecognizer(backend)
        result = await recognizer._find_entity_co_occurrences(
            "technology", min_occurrences=2
        )

        assert len(result) == 2
        assert all(isinstance(p, Pattern) for p in result)
        assert result[0].name == "Co-occurrence: Python + pytest"
        assert result[0].occurrences == 5
        assert result[1].name == "Co-occurrence: FastAPI + async"
