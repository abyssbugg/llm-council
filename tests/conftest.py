"""Pytest configuration and fixtures for LLM Council tests."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any
import httpx


# Test data fixtures
@pytest.fixture
def sample_messages() -> List[Dict[str, str]]:
    """Sample message list for testing."""
    return [
        {"role": "user", "content": "What is the capital of France?"}
    ]


@pytest.fixture
def sample_openrouter_response() -> Dict[str, Any]:
    """Sample OpenRouter API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "The capital of France is Paris.",
                    "role": "assistant"
                }
            }
        ]
    }


@pytest.fixture
def sample_openrouter_response_with_reasoning() -> Dict[str, Any]:
    """Sample OpenRouter API response with reasoning details."""
    return {
        "choices": [
            {
                "message": {
                    "content": "The capital of France is Paris.",
                    "role": "assistant",
                    "reasoning_details": "Paris has been the capital since..."
                }
            }
        ]
    }


@pytest.fixture
def sample_stage1_results() -> List[Dict[str, Any]]:
    """Sample Stage 1 results."""
    return [
        {
            "model": "openai/gpt-4o",
            "response": "Response from GPT-4"
        },
        {
            "model": "anthropic/claude-3-opus",
            "response": "Response from Claude"
        }
    ]


@pytest.fixture
def sample_stage2_results() -> List[Dict[str, Any]]:
    """Sample Stage 2 results."""
    return [
        {
            "model": "openai/gpt-4o",
            "ranking": "1. Response A, 2. Response B",
            "parsed_ranking": ["Response A", "Response B"]
        }
    ]


@pytest.fixture
def sample_stage3_result() -> Dict[str, Any]:
    """Sample Stage 3 result."""
    return {
        "model": "google/gemini-pro",
        "response": "Final synthesized answer..."
    }


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for API testing."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Test response", "role": "assistant"}}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_openrouter_api_key(monkeypatch):
    """Mock OpenRouter API key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key_12345")


@pytest.fixture
def mock_chutes_ai_api_key(monkeypatch):
    """Mock Chutes AI API key."""
    monkeypatch.setenv("CHUTES_AI_API_KEY", "test_chutes_key")


@pytest.fixture
def mock_huggingface_api_key(monkeypatch):
    """Mock Hugging Face API key."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test_hf_key")


# Async test fixtures
@pytest.fixture
async def async_mock_httpx():
    """Async version of httpx client mock."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Test response", "role": "assistant"}}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client
        yield mock_client
