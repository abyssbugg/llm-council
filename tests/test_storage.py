"""Tests for backend/storage.py module."""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime
import uuid


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data" / "conversations"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def mock_storage_dir(temp_data_dir):
    """Mock the storage directory path."""
    with patch('backend.storage.DATA_DIR', str(temp_data_dir)):
        yield str(temp_data_dir)


@pytest.fixture
def sample_conversation():
    """Sample conversation data."""
    return {
        "id": str(uuid.uuid4()),
        "created_at": "2024-01-01T00:00:00",
        "title": "Test Conversation",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            },
            {
                "role": "assistant",
                "stage1": [{"model": "gpt-4", "response": "Hi"}],
                "stage2": [],
                "stage3": {"model": "chairman", "response": "Hello there!"}
            }
        ]
    }


class TestStorageModule:
    """Test storage module functions."""

    def test_list_conversations_empty(self, mock_storage_dir):
        """List conversations when directory is empty."""
        from backend import storage
        result = storage.list_conversations()
        assert result == []

    def test_list_conversations_with_data(self, mock_storage_dir, sample_conversation, temp_data_dir):
        """List conversations with existing data."""
        # Create sample conversation files
        conv1 = sample_conversation.copy()
        conv1["id"] = "conv1"
        conv2 = sample_conversation.copy()
        conv2["id"] = "conv2"

        with open(temp_data_dir / "conv1.json", "w") as f:
            json.dump(conv1, f)
        with open(temp_data_dir / "conv2.json", "w") as f:
            json.dump(conv2, f)

        from backend import storage
        result = storage.list_conversations()

        assert len(result) == 2
        ids = [c["id"] for c in result]
        assert "conv1" in ids
        assert "conv2" in ids

    def test_create_conversation(self, mock_storage_dir):
        """Create a new conversation."""
        from backend import storage
        conversation_id = str(uuid.uuid4())

        result = storage.create_conversation(conversation_id)

        assert result["id"] == conversation_id
        assert result["title"] == "New Conversation"
        assert result["messages"] == []
        assert "created_at" in result

        # Verify file was created
        from backend.storage import DATA_DIR
        file_path = Path(DATA_DIR) / f"{conversation_id}.json"
        assert file_path.exists()

    def test_get_conversation_exists(self, mock_storage_dir, sample_conversation, temp_data_dir):
        """Get an existing conversation."""
        conv_id = sample_conversation["id"]
        with open(temp_data_dir / f"{conv_id}.json", "w") as f:
            json.dump(sample_conversation, f)

        from backend import storage
        result = storage.get_conversation(conv_id)

        assert result is not None
        assert result["id"] == conv_id
        assert result["title"] == "Test Conversation"

    def test_get_conversation_not_exists(self, mock_storage_dir):
        """Get a conversation that doesn't exist."""
        from backend import storage
        result = storage.get_conversation("nonexistent-id")
        assert result is None

    def test_add_user_message(self, mock_storage_dir, temp_data_dir):
        """Add a user message to conversation."""
        from backend import storage
        conv_id = str(uuid.uuid4())
        storage.create_conversation(conv_id)

        storage.add_user_message(conv_id, "Test message")

        # Verify message was added
        result = storage.get_conversation(conv_id)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Test message"

    def test_add_assistant_message(self, mock_storage_dir, temp_data_dir):
        """Add an assistant message to conversation."""
        from backend import storage
        conv_id = str(uuid.uuid4())
        storage.create_conversation(conv_id)

        stage1 = [{"model": "gpt-4", "response": "Test"}]
        stage2 = [{"model": "model1", "ranking": "Ranking"}]
        stage3 = {"model": "chairman", "response": "Final"}

        storage.add_assistant_message(conv_id, stage1, stage2, stage3)

        # Verify message was added
        result = storage.get_conversation(conv_id)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"
        assert result["messages"][0]["stage1"] == stage1
        assert result["messages"][0]["stage2"] == stage2
        assert result["messages"][0]["stage3"] == stage3

    def test_update_conversation_title(self, mock_storage_dir, temp_data_dir):
        """Update conversation title."""
        from backend import storage
        conv_id = str(uuid.uuid4())
        storage.create_conversation(conv_id)

        storage.update_conversation_title(conv_id, "Updated Title")

        # Verify title was updated
        result = storage.get_conversation(conv_id)
        assert result["title"] == "Updated Title"

    def test_conversation_persistence(self, mock_storage_dir, temp_data_dir):
        """Verify conversations persist across module reloads.

        Note: This test verifies that data is written to disk correctly.
        Module reload testing with mocks is complex, so we verify
        file persistence instead.
        """
        from backend import storage
        conv_id = str(uuid.uuid4())
        storage.create_conversation(conv_id)
        storage.add_user_message(conv_id, "Test")

        # Verify file was created and contains data
        file_path = temp_data_dir / f"{conv_id}.json"
        assert file_path.exists()

        # Read file directly to verify persistence
        import json
        with open(file_path, "r") as f:
            data = json.load(f)

        assert data["id"] == conv_id
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Test"

    def test_json_file_format(self, mock_storage_dir, temp_data_dir):
        """Verify JSON file format is correct."""
        from backend import storage
        conv_id = str(uuid.uuid4())
        storage.create_conversation(conv_id)
        storage.add_user_message(conv_id, "Test")

        # Read the file directly
        file_path = temp_data_dir / f"{conv_id}.json"
        with open(file_path, "r") as f:
            data = json.load(f)

        assert "id" in data
        assert "created_at" in data
        assert "title" in data
        assert "messages" in data
        assert isinstance(data["messages"], list)
