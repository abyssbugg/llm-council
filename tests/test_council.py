"""Tests for backend/council.py module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.council import (
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
    calculate_aggregate_rankings,
    parse_ranking_from_text,
    run_full_council,
)


class TestParseRankingFromText:
    """Test ranking extraction from model responses."""

    def test_parse_ranking_numbered_list(self):
        """Extract ranking from numbered list format."""
        text = """
        I think Response A is best because...
        FINAL RANKING:
        1. Response A
        2. Response C
        3. Response B
        """
        result = parse_ranking_from_text(text)
        assert result == ["Response A", "Response C", "Response B"]

    def test_parse_ranking_plain_format(self):
        """Extract ranking from plain format without numbers."""
        text = """
        My evaluation is complete.
        FINAL RANKING: Response B, Response A, Response C
        """
        result = parse_ranking_from_text(text)
        assert result == ["Response B", "Response A", "Response C"]

    def test_parse_ranking_no_final_section(self):
        """Returns empty list when no FINAL RANKING section found."""
        text = "This is just some text without any ranking."
        result = parse_ranking_from_text(text)
        assert result == []

    def test_parse_ranking_with_extra_text(self):
        """Extract ranking even with extra text after."""
        text = """
        FINAL RANKING:
        1. Response A
        2. Response B
        Thanks for asking!
        """
        result = parse_ranking_from_text(text)
        assert result == ["Response A", "Response B"]


class TestCalculateAggregateRankings:
    """Test aggregate ranking calculation."""

    def test_calculate_aggregate_single_vote(self):
        """Calculate rankings with single voter."""
        stage2_results = [
            {
                "model": "model1",
                "ranking": "FINAL RANKING:\n1. Response A\n2. Response B\n3. Response C"
            }
        ]
        label_to_model = {"Response A": "gpt-4", "Response B": "claude", "Response C": "gemini"}

        result = calculate_aggregate_rankings(stage2_results, label_to_model)

        assert result[0]["model"] == "gpt-4"
        assert result[0]["average_rank"] == 1.0
        assert result[0]["rankings_count"] == 1

    def test_calculate_aggregate_multiple_votes(self):
        """Calculate rankings with multiple voters."""
        stage2_results = [
            {
                "model": "model1",
                "ranking": "FINAL RANKING:\n1. Response A\n2. Response B\n3. Response C"
            },
            {
                "model": "model2",
                "ranking": "FINAL RANKING:\n1. Response B\n2. Response A\n3. Response C"
            }
        ]
        label_to_model = {"Response A": "gpt-4", "Response B": "claude", "Response C": "gemini"}

        result = calculate_aggregate_rankings(stage2_results, label_to_model)

        # Response A: positions 1 + 2 = 3 / 2 = 1.5
        # Response B: positions 2 + 1 = 3 / 2 = 1.5
        # Response C: positions 3 + 3 = 6 / 2 = 3.0
        assert len(result) == 3
        models = [r["model"] for r in result]
        assert "gpt-4" in models
        assert "claude" in models
        assert "gemini" in models
        # First two should have rank 1.5, last should be 3.0
        assert result[0]["average_rank"] == 1.5
        assert result[1]["average_rank"] == 1.5
        assert result[2]["average_rank"] == 3.0

    def test_calculate_aggregate_empty_results(self):
        """Handle empty stage2 results."""
        result = calculate_aggregate_rankings([], {})
        assert result == []


@pytest.mark.asyncio
class TestStage1CollectResponses:
    """Test Stage 1: Collect individual model responses."""

    @pytest.mark.asyncio
    async def test_stage1_success(self):
        """Successfully collect responses from all models."""
        with patch('backend.council.query_models_parallel') as mock_query:
            mock_query.return_value = {
                "model1": {"content": "Response 1", "reasoning_details": None},
                "model2": {"content": "Response 2", "reasoning_details": None},
            }

            result = await stage1_collect_responses("Test query")

            assert len(result) == 2
            assert result[0]["model"] in ["model1", "model2"]
            assert result[0]["response"] in ["Response 1", "Response 2"]

    @pytest.mark.asyncio
    async def test_stage1_partial_failure(self):
        """Continue when some models fail."""
        with patch('backend.council.query_models_parallel') as mock_query:
            mock_query.return_value = {
                "model1": {"content": "Response 1", "reasoning_details": None},
                "model2": None,  # Failed model
            }

            result = await stage1_collect_responses("Test query")

            # Should only return successful responses
            assert len(result) == 1
            assert result[0]["model"] == "model1"


@pytest.mark.asyncio
class TestStage2CollectRankings:
    """Test Stage 2: Collect peer rankings."""

    @pytest.mark.asyncio
    async def test_stage2_success(self):
        """Successfully collect rankings with anonymization."""
        stage1_results = [
            {"model": "gpt-4", "response": "Response A"},
            {"model": "claude", "response": "Response B"},
        ]

        with patch('backend.council.query_models_parallel') as mock_query:
            mock_query.return_value = {
                "model1": {
                    "content": "FINAL RANKING:\n1. Response A\n2. Response B",
                    "reasoning_details": None
                }
            }

            rankings, label_to_model = await stage2_collect_rankings("Test query", stage1_results)

            assert len(rankings) == 1
            assert isinstance(label_to_model, dict)
            assert len(label_to_model) == 2  # Response A, Response B

    @pytest.mark.asyncio
    async def test_stage2_empty_stage1(self):
        """Handle empty stage1 results."""
        with patch('backend.council.query_models_parallel') as mock_query:
            mock_query.return_value = {}

            rankings, label_to_model = await stage2_collect_rankings("Test query", [])

            assert rankings == []
            assert label_to_model == {}


@pytest.mark.asyncio
class TestStage3SynthesizeFinal:
    """Test Stage 3: Final synthesis."""

    @pytest.mark.asyncio
    async def test_stage3_success(self):
        """Successfully synthesize final answer."""
        stage1_results = [
            {"model": "gpt-4", "response": "Response A"},
        ]
        stage2_results = [
            {"model": "model1", "ranking": "Ranking text"}
        ]

        with patch('backend.council.query_model') as mock_query, \
             patch('backend.council.CHAIRMAN_MODEL', 'test-chairman-model'):
            mock_query.return_value = {
                "content": "Final synthesized answer",
                "reasoning_details": None
            }

            result = await stage3_synthesize_final("Test query", stage1_results, stage2_results)

            assert result["model"] == "test-chairman-model"
            assert result["response"] == "Final synthesized answer"


@pytest.mark.asyncio
class TestRunFullCouncil:
    """Test the full 3-stage council process."""

    @pytest.mark.asyncio
    async def test_run_full_council_success(self):
        """Successfully run all three stages."""
        with patch('backend.council.stage1_collect_responses') as mock_stage1, \
             patch('backend.council.stage2_collect_rankings') as mock_stage2, \
             patch('backend.council.stage3_synthesize_final') as mock_stage3, \
             patch('backend.council.calculate_aggregate_rankings') as mock_aggregate:

            mock_stage1.return_value = [{"model": "gpt-4", "response": "Response A"}]
            mock_stage2.return_value = ([{"model": "model1", "ranking": "Ranking"}], {"label_to_model": {"Response A": "gpt-4"}})
            mock_aggregate.return_value = []
            mock_stage3.return_value = {"model": "chairman", "response": "Final answer"}

            stage1, stage2, stage3, metadata = await run_full_council("Test query")

            assert len(stage1) == 1
            assert len(stage2) == 1
            assert stage3["response"] == "Final answer"
            assert "label_to_model" in metadata
