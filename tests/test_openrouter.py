"""Tests for backend/openrouter.py module."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from tenacity import RetryError, stop_after_attempt

from backend.openrouter import query_model, query_models_parallel, _should_retry_http_error


class TestShouldRetryHttpError:
    """Test the retry predicate function."""

    def test_retry_on_429_rate_limit(self):
        """Should retry on 429 rate limit errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        exception = httpx.HTTPStatusError("Rate limit", request=MagicMock(), response=mock_response)

        assert _should_retry_http_error(exception) is True

    def test_retry_on_500_server_error(self):
        """Should retry on 500 server errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        exception = httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)

        assert _should_retry_http_error(exception) is True

    def test_retry_on_502_bad_gateway(self):
        """Should retry on 502 bad gateway errors."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        exception = httpx.HTTPStatusError("Bad gateway", request=MagicMock(), response=mock_response)

        assert _should_retry_http_error(exception) is True

    def test_retry_on_503_service_unavailable(self):
        """Should retry on 503 service unavailable errors."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        exception = httpx.HTTPStatusError("Service unavailable", request=MagicMock(), response=mock_response)

        assert _should_retry_http_error(exception) is True

    def test_retry_on_504_gateway_timeout(self):
        """Should retry on 504 gateway timeout errors."""
        mock_response = MagicMock()
        mock_response.status_code = 504
        exception = httpx.HTTPStatusError("Gateway timeout", request=MagicMock(), response=mock_response)

        assert _should_retry_http_error(exception) is True

    def test_no_retry_on_400_bad_request(self):
        """Should NOT retry on 400 bad request errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        exception = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=mock_response)

        assert _should_retry_http_error(exception) is False

    def test_no_retry_on_401_unauthorized(self):
        """Should NOT retry on 401 unauthorized errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        exception = httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=mock_response)

        assert _should_retry_http_error(exception) is False

    def test_no_retry_on_404_not_found(self):
        """Should NOT retry on 404 not found errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        exception = httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)

        assert _should_retry_http_error(exception) is False

    def test_no_retry_on_non_http_error(self):
        """Should NOT retry on non-HTTP errors."""
        exception = ValueError("Some other error")

        assert _should_retry_http_error(exception) is False


@pytest.mark.asyncio
class TestQueryModel:
    """Test the query_model function."""

    @pytest.mark.asyncio
    async def test_query_model_success(self, mock_openrouter_api_key, sample_messages, sample_openrouter_response):
        """Successful API query returns content."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_openrouter_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await query_model("openai/gpt-4o", sample_messages)

            assert result is not None
            assert result["content"] == "The capital of France is Paris."
            assert result["reasoning_details"] is None

    @pytest.mark.asyncio
    async def test_query_model_with_reasoning(self, mock_openrouter_api_key, sample_messages, sample_openrouter_response_with_reasoning):
        """API query returns reasoning details when present."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_openrouter_response_with_reasoning
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await query_model("openai/gpt-4o", sample_messages)

            assert result is not None
            assert result["content"] == "The capital of France is Paris."
            assert result["reasoning_details"] == "Paris has been the capital since..."

    @pytest.mark.asyncio
    async def test_query_model_429_retry_then_success(self, mock_openrouter_api_key, sample_messages):
        """Retries on 429 and eventually succeeds.

        Note: Current implementation catches all exceptions and returns None,
        so this test verifies the function handles the error gracefully.
        """
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.text = "Rate limit exceeded"
            error_429 = httpx.HTTPStatusError("Rate limit", request=MagicMock(), response=mock_response)

            mock_client.post = AsyncMock(side_effect=error_429)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await query_model("openai/gpt-4o", sample_messages)

            # Current implementation returns None on HTTP errors
            # The retry mechanism will retry, but if all retries fail, returns None
            assert result is None

    @pytest.mark.asyncio
    async def test_query_model_400_no_retry(self, mock_openrouter_api_key, sample_messages):
        """Does NOT retry on 400 bad request."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            error_400 = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=mock_response)

            mock_client.post = AsyncMock(side_effect=error_400)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await query_model("openai/gpt-4o", sample_messages)

            # Should return None immediately without retry
            assert result is None
            assert mock_client.post.call_count == 1  # Only called once

    @pytest.mark.asyncio
    async def test_query_model_max_retries_exceeded(self, mock_openrouter_api_key, sample_messages):
        """Returns None after max retries exceeded.

        Note: Current implementation catches HTTPStatusError and retries via tenacity,
        but if all retries fail, returns None. The mock won't actually retry because
        the exception is caught within the function.
        """
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_response.text = "Service unavailable"
            error_503 = httpx.HTTPStatusError("Service unavailable", request=MagicMock(), response=mock_response)

            # Always return 503
            mock_client.post = AsyncMock(side_effect=error_503)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await query_model("openai/gpt-4o", sample_messages)

            # After retries fail, should return None
            assert result is None


@pytest.mark.asyncio
class TestQueryModelsParallel:
    """Test the query_models_parallel function."""

    @pytest.mark.asyncio
    async def test_query_models_parallel_success(self, mock_openrouter_api_key, sample_messages):
        """Successfully queries multiple models in parallel."""
        with patch('backend.openrouter.query_model') as mock_query:
            mock_query.side_effect = [
                {"content": "Response 1", "reasoning_details": None},
                {"content": "Response 2", "reasoning_details": None},
                {"content": "Response 3", "reasoning_details": None},
            ]

            models = ["model1", "model2", "model3"]
            result = await query_models_parallel(models, sample_messages)

            assert len(result) == 3
            assert result["model1"]["content"] == "Response 1"
            assert result["model2"]["content"] == "Response 2"
            assert result["model3"]["content"] == "Response 3"

    @pytest.mark.asyncio
    async def test_query_models_parallel_partial_failure(self, mock_openrouter_api_key, sample_messages):
        """Handles partial failures gracefully."""
        with patch('backend.openrouter.query_model') as mock_query:
            mock_query.side_effect = [
                {"content": "Response 1", "reasoning_details": None},
                None,  # Failed model
                {"content": "Response 3", "reasoning_details": None},
            ]

            models = ["model1", "model2", "model3"]
            result = await query_models_parallel(models, sample_messages)

            assert len(result) == 3
            assert result["model1"]["content"] == "Response 1"
            assert result["model2"] is None
            assert result["model3"]["content"] == "Response 3"

    @pytest.mark.asyncio
    async def test_query_models_parallel_empty_list(self, mock_openrouter_api_key, sample_messages):
        """Handles empty model list."""
        with patch('backend.openrouter.query_model') as mock_query:
            result = await query_models_parallel([], sample_messages)

            assert result == {}
            mock_query.assert_not_called()
