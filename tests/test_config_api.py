"""Tests for the configuration API module."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from backend.main import app
from backend.models import Base, User, UserConfig, Message
from backend.database import get_db
from backend.auth import create_access_token


# Test database URL (in-memory SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_engine():
    """Create a test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_session(test_engine):
    """Create a test database session."""
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session() as session:
        yield session


@pytest.fixture
async def test_user(db_session):
    """Create a test user."""
    user = User(
        email="config@example.com",
        hashed_password="hashed_password_here",
        display_name="Test User",
        daily_budget_limit=20.0,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def authenticated_user(test_user):
    """Create access token for test user."""
    return create_access_token(data={"sub": test_user.email, "user_id": test_user.id})


@pytest.fixture
def client(db_session):
    """Create a test client with database dependency override."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestConfigEndpoints:
    """Tests for configuration API endpoints."""

    @pytest.mark.asyncio
    async def test_get_config_unauthorized(self, client):
        """Test getting config without authentication."""
        response = client.get("/api/config/")
        assert response.status_code == 403  # FastAPI OAuth2 returns 403, not 401

    @pytest.mark.asyncio
    async def test_get_config_with_default(self, client, authenticated_user):
        """Test getting config creates default."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}
        response = client.get("/api/config/", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert "council" in data
        assert "generation" in data
        assert "context" in data

        # Check default values
        assert data["council"]["models"] == [
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro",
        ]
        assert data["council"]["chairman_model"] == "google/gemini-pro"
        assert data["generation"]["temperature"] == 0.7
        assert data["generation"]["max_tokens"] == 4096
        assert data["context"]["max_history_messages"] == 10

    @pytest.mark.asyncio
    async def test_get_council_config(self, client, authenticated_user):
        """Test getting council configuration."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}
        response = client.get("/api/config/council", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "chairman_model" in data
        assert isinstance(data["models"], list)

    @pytest.mark.asyncio
    async def test_update_council_config(self, client, authenticated_user):
        """Test updating council configuration."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}

        new_models = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]
        update_data = {
            "models": new_models,
            "chairman_model": "openai/gpt-4o-mini"
        }

        response = client.put("/api/config/council", json=update_data, headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert data["models"] == new_models
        assert data["chairman_model"] == "openai/gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_update_council_config_empty_models(self, client, authenticated_user):
        """Test updating council config with empty models fails."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}

        update_data = {
            "models": [],
            "chairman_model": "openai/gpt-4o"
        }

        response = client.put("/api/config/council", json=update_data, headers=headers)

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_get_generation_params(self, client, authenticated_user):
        """Test getting generation parameters."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}
        response = client.get("/api/config/generation", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert "temperature" in data
        assert "max_tokens" in data
        assert "system_prompt" in data
        assert data["temperature"] == 0.7
        assert data["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_update_generation_params(self, client, authenticated_user):
        """Test updating generation parameters."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}

        update_data = {
            "temperature": 0.5,
            "max_tokens": 2048,
            "system_prompt": "You are a helpful assistant."
        }

        response = client.put("/api/config/generation", json=update_data, headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert data["temperature"] == 0.5
        assert data["max_tokens"] == 2048
        assert data["system_prompt"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_get_context_config(self, client, authenticated_user):
        """Test getting context configuration."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}
        response = client.get("/api/config/context", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert "max_history_messages" in data
        assert data["max_history_messages"] == 10

    @pytest.mark.asyncio
    async def test_update_context_config(self, client, authenticated_user):
        """Test updating context configuration."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}

        update_data = {"max_history_messages": 20}

        response = client.put("/api/config/context", json=update_data, headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert data["max_history_messages"] == 20

    @pytest.mark.asyncio
    async def test_reset_config(self, client, authenticated_user):
        """Test resetting configuration to defaults."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}

        # First modify config
        update_data = {"temperature": 0.1, "max_tokens": 1000}
        client.put("/api/config/generation", json=update_data, headers=headers)

        # Reset
        response = client.post("/api/config/reset", headers=headers)

        assert response.status_code == 200
        data = response.json()

        # Should be back to defaults
        assert data["generation"]["temperature"] == 0.7
        assert data["generation"]["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_update_full_config(self, client, authenticated_user):
        """Test updating full configuration."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}

        update_data = {
            "council_models": ["openai/gpt-4o"],
            "chairman_model": "openai/gpt-4o",
            "temperature": 0.8,
            "max_tokens": 8192,
            "system_prompt": "Custom prompt",
            "max_history_messages": 5,
        }

        response = client.put("/api/config/", json=update_data, headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert data["council"]["models"] == ["openai/gpt-4o"]
        assert data["council"]["chairman_model"] == "openai/gpt-4o"
        assert data["generation"]["temperature"] == 0.8
        assert data["generation"]["max_tokens"] == 8192
        assert data["generation"]["system_prompt"] == "Custom prompt"
        assert data["context"]["max_history_messages"] == 5


class TestContextManagement:
    """Tests for context management service."""

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(self, db_session, test_user):
        """Test getting history for conversation with no messages."""
        from backend.context import get_context_service
        from backend.models import Conversation

        # Create conversation
        conversation = Conversation(
            id="test-conv-history",
            user_id=test_user.id,
            title="Test History",
        )
        db_session.add(conversation)
        await db_session.commit()

        # Get history
        context_service = get_context_service()
        history = await context_service.get_conversation_history(
            "test-conv-history",
            test_user.id,
            db=db_session
        )

        assert history == []

    @pytest.mark.asyncio
    async def test_get_conversation_history_with_messages(self, db_session, test_user):
        """Test getting history for conversation with messages."""
        from backend.context import get_context_service
        from backend.models import Conversation, Message

        # Create conversation with messages
        conversation = Conversation(
            id="test-conv-with-msgs",
            user_id=test_user.id,
            title="Test Messages",
        )
        db_session.add(conversation)
        await db_session.flush()

        # Add messages
        for i in range(5):
            msg = Message(
                conversation_id="test-conv-with-msgs",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
            )
            db_session.add(msg)

        await db_session.commit()

        # Get history
        context_service = get_context_service()
        history = await context_service.get_conversation_history(
            "test-conv-with-msgs",
            test_user.id,
            db=db_session
        )

        assert len(history) == 5
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Message 0"

    @pytest.mark.asyncio
    async def test_get_conversation_history_max_messages(self, db_session, test_user):
        """Test getting history respects max_messages limit."""
        from backend.context import get_context_service
        from backend.models import Conversation, Message

        # Create conversation with more messages than default max
        conversation = Conversation(
            id="test-conv-max",
            user_id=test_user.id,
            title="Test Max Messages",
        )
        db_session.add(conversation)
        await db_session.flush()

        # Add 15 messages
        for i in range(15):
            msg = Message(
                conversation_id="test-conv-max",
                role="user",
                content=f"Message {i}",
            )
            db_session.add(msg)

        await db_session.commit()

        # Get history with max_messages=5
        context_service = get_context_service()
        history = await context_service.get_conversation_history(
            "test-conv-max",
            test_user.id,
            max_messages=5,
            db=db_session
        )

        # Should return last 5 messages
        assert len(history) == 5
        assert history[0]["content"] == "Message 10"
        assert history[4]["content"] == "Message 14"

    @pytest.mark.asyncio
    async def test_get_context_window_info(self, db_session, test_user):
        """Test getting context window information."""
        from backend.context import get_context_service
        from backend.models import Conversation, Message

        # Create conversation
        conversation = Conversation(
            id="test-context-window",
            user_id=test_user.id,
            title="Context Window Test",
        )
        db_session.add(conversation)
        await db_session.flush()

        # Add some messages
        for i in range(15):
            msg = Message(
                conversation_id="test-context-window",
                role="user",
                content=f"Message {i} with some content",
            )
            db_session.add(msg)

        await db_session.commit()

        # Get context info
        context_service = get_context_service()
        info = await context_service.get_context_window_info(
            "test-context-window",
            test_user.id,
            db=db_session
        )

        assert info["total_messages"] == 15
        assert info["context_messages"] == 10  # Default max
        assert info["max_messages"] == 10
        assert info["truncated_messages"] == 5
        assert info["estimated_tokens"] > 0

    @pytest.mark.asyncio
    async def test_truncate_conversation(self, db_session, test_user):
        """Test truncating a conversation."""
        from backend.context import get_context_service
        from backend.models import Conversation, Message

        # Create conversation
        conversation = Conversation(
            id="test-truncate",
            user_id=test_user.id,
            title="Truncate Test",
        )
        db_session.add(conversation)
        await db_session.flush()

        # Add 10 messages
        for i in range(10):
            msg = Message(
                conversation_id="test-truncate",
                role="user",
                content=f"Message {i}",
            )
            db_session.add(msg)

        await db_session.commit()

        # Truncate to 5 messages
        context_service = get_context_service()
        deleted = await context_service.truncate_conversation(
            "test-truncate",
            test_user.id,
            keep_messages=5,
            db=db_session
        )

        assert deleted == 5

        # Verify remaining messages
        result = await db_session.execute(
            select(Message).where(Message.conversation_id == "test-truncate")
        )
        remaining = result.scalars().all()

        assert len(remaining) == 5

    @pytest.mark.asyncio
    async def test_truncate_conversation_nothing_to_delete(self, db_session, test_user):
        """Test truncating when keeping more than exists."""
        from backend.context import get_context_service
        from backend.models import Conversation, Message

        # Create conversation with 3 messages
        conversation = Conversation(
            id="test-no-truncate",
            user_id=test_user.id,
            title="No Truncate Test",
        )
        db_session.add(conversation)
        await db_session.flush()

        for i in range(3):
            msg = Message(
                conversation_id="test-no-truncate",
                role="user",
                content=f"Message {i}",
            )
            db_session.add(msg)

        await db_session.commit()

        # Try to keep 10 (more than exists)
        context_service = get_context_service()
        deleted = await context_service.truncate_conversation(
            "test-no-truncate",
            test_user.id,
            keep_messages=10,
            db=db_session
        )

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_get_conversation_history_not_found(self, db_session, test_user):
        """Test getting history for non-existent conversation."""
        from backend.context import get_context_service

        context_service = get_context_service()

        with pytest.raises(ValueError, match="not found or access denied"):
            await context_service.get_conversation_history(
                "non-existent-conv",
                test_user.id,
                db=db_session
            )

    @pytest.mark.asyncio
    async def test_get_conversation_history_wrong_user(self, db_session, test_user):
        """Test getting history for conversation owned by another user."""
        from backend.context import get_context_service
        from backend.models import User, Conversation

        # Create another user
        other_user = User(
            email="other@example.com",
            hashed_password="hash",
        )
        db_session.add(other_user)
        await db_session.flush()

        # Create conversation for other user
        conversation = Conversation(
            id="other-users-conv",
            user_id=other_user.id,
            title="Other User's Conv",
        )
        db_session.add(conversation)
        await db_session.commit()

        # Try to get history with different user
        context_service = get_context_service()

        with pytest.raises(ValueError, match="not found or access denied"):
            await context_service.get_conversation_history(
                "other-users-conv",
                test_user.id,
                db=db_session
            )


class TestGlobalContextService:
    """Tests for global context service functions."""

    def test_get_context_service_singleton(self):
        """Test that get_context_service returns a singleton."""
        from backend.context import get_context_service, reset_context_service

        reset_context_service()

        service1 = get_context_service()
        service2 = get_context_service()

        assert service1 is service2

        reset_context_service()

    def test_reset_context_service(self):
        """Test resetting the global service."""
        from backend.context import get_context_service, reset_context_service

        service1 = get_context_service()

        reset_context_service()

        service2 = get_context_service()

        # Should be a new instance
        assert service1 is not service2

        reset_context_service()
