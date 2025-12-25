"""Tests for the export API and service."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from backend.main import app
from backend.models import Base, User, Conversation, Message
from backend.database import get_db
from backend.auth import create_access_token
from backend.export import get_export_service, reset_export_service


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
        email="export@example.com",
        hashed_password="hashed_password_here",
        display_name="Export User",
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


class TestExportService:
    """Tests for the export service."""

    @pytest.mark.asyncio
    async def test_export_conversation_to_markdown_empty(self, db_session, test_user):
        """Test exporting empty conversation to markdown."""
        conversation = Conversation(
            id="export-empty-conv",
            user_id=test_user.id,
            title="Empty Export Test",
        )
        db_session.add(conversation)
        await db_session.commit()

        export_service = get_export_service()
        markdown = await export_service.export_conversation_to_markdown(
            "export-empty-conv",
            test_user.id,
            db_session,
        )

        assert "# Empty Export Test" in markdown
        assert "**Created:**" in markdown

    @pytest.mark.asyncio
    async def test_export_conversation_to_markdown_with_messages(self, db_session, test_user):
        """Test exporting conversation with messages."""
        conversation = Conversation(
            id="export-msgs-conv",
            user_id=test_user.id,
            title="Messages Export Test",
        )
        db_session.add(conversation)
        await db_session.flush()

        # Add user message
        user_msg = Message(
            conversation_id="export-msgs-conv",
            role="user",
            content="Hello, this is a test message.",
        )
        db_session.add(user_msg)

        # Add assistant message with stages
        assistant_msg = Message(
            conversation_id="export-msgs-conv",
            role="assistant",
            content="This is the assistant's response.",
            stage1={
                "responses": {
                    "openai/gpt-4o": {"content": "GPT-4 response"},
                    "anthropic/claude": {"content": "Claude response"},
                }
            },
            stage2={
                "rankings": [
                    {"model": "openai/gpt-4o", "ranking": [1, 2]},
                ]
            },
        )
        db_session.add(assistant_msg)

        await db_session.commit()

        export_service = get_export_service()
        markdown = await export_service.export_conversation_to_markdown(
            "export-msgs-conv",
            test_user.id,
            db_session,
            include_metadata=True,
        )

        assert "# Messages Export Test" in markdown
        assert "## 👤 User" in markdown
        assert "Hello, this is a test message." in markdown
        assert "## 🤖 Assistant" in markdown
        assert "This is the assistant's response." in markdown
        assert "### Stage 1: Individual Council Responses" in markdown

    @pytest.mark.asyncio
    async def test_export_conversation_to_dict(self, db_session, test_user):
        """Test exporting conversation to structured dict."""
        conversation = Conversation(
            id="export-dict-conv",
            user_id=test_user.id,
            title="Dict Export Test",
        )
        db_session.add(conversation)
        await db_session.flush()

        msg = Message(
            conversation_id="export-dict-conv",
            role="user",
            content="Test message",
        )
        db_session.add(msg)
        await db_session.commit()

        export_service = get_export_service()
        data = await export_service.export_conversation_to_dict(
            "export-dict-conv",
            test_user.id,
            db_session,
        )

        assert data["id"] == "export-dict-conv"
        assert data["title"] == "Dict Export Test"
        assert "messages" in data
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Test message"

    @pytest.mark.asyncio
    async def test_export_conversation_not_found(self, db_session, test_user):
        """Test exporting non-existent conversation."""
        export_service = get_export_service()

        with pytest.raises(ValueError, match="not found or access denied"):
            await export_service.export_conversation_to_markdown(
                "non-existent",
                test_user.id,
                db_session,
            )

    @pytest.mark.asyncio
    async def test_export_conversation_wrong_user(self, db_session, test_user):
        """Test exporting conversation owned by another user."""
        # Create another user
        other_user = User(
            email="other@example.com",
            hashed_password="hash",
        )
        db_session.add(other_user)
        await db_session.flush()

        # Create conversation for other user
        conversation = Conversation(
            id="other-export-conv",
            user_id=other_user.id,
            title="Other User's Export",
        )
        db_session.add(conversation)
        await db_session.commit()

        export_service = get_export_service()

        with pytest.raises(ValueError, match="not found or access denied"):
            await export_service.export_conversation_to_markdown(
                "other-export-conv",
                test_user.id,
                db_session,
            )

    @pytest.mark.asyncio
    async def test_get_export_summary(self, db_session, test_user):
        """Test getting export summary."""
        # Create some conversations
        for i in range(3):
            conv = Conversation(
                id=f"export-summary-{i}",
                user_id=test_user.id,
                title=f"Summary Test {i}",
            )
            db_session.add(conv)
        await db_session.commit()

        export_service = get_export_service()
        summary = await export_service.get_export_summary(test_user.id, db_session)

        assert summary["total_conversations"] == 3
        assert len(summary["conversations"]) == 3

    @pytest.mark.asyncio
    async def test_export_multiple_conversations(self, db_session, test_user):
        """Test exporting multiple conversations."""
        # Create conversations
        conv_ids = []
        for i in range(2):
            conv = Conversation(
                id=f"multi-export-{i}",
                user_id=test_user.id,
                title=f"Multi Export {i}",
            )
            db_session.add(conv)
            await db_session.flush()

            msg = Message(
                conversation_id=f"multi-export-{i}",
                role="user",
                content=f"Message {i}",
            )
            db_session.add(msg)
            conv_ids.append(f"multi-export-{i}")

        await db_session.commit()

        export_service = get_export_service()
        markdown = await export_service.export_multiple_conversations_to_markdown(
            conv_ids,
            test_user.id,
            db_session,
        )

        assert "# LLM Council Conversations Export" in markdown
        assert "# Multi Export 0" in markdown
        assert "# Multi Export 1" in markdown


class TestExportAPI:
    """Tests for the export API endpoints."""

    @pytest.mark.asyncio
    async def test_get_export_summary_unauthorized(self, client):
        """Test getting export summary without authentication."""
        response = client.get("/api/export/summary")
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_export_summary_success(self, client, authenticated_user, db_session, test_user):
        """Test getting export summary."""
        # Create a conversation
        conversation = Conversation(
            id="api-export-summary-conv",
            user_id=test_user.id,
            title="API Export Summary Test",
        )
        db_session.add(conversation)
        await db_session.commit()

        headers = {"Authorization": f"Bearer {authenticated_user}"}
        response = client.get("/api/export/summary", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "total_conversations" in data

    @pytest.mark.asyncio
    async def test_export_conversation_markdown(self, client, authenticated_user, db_session, test_user):
        """Test exporting conversation to markdown via API."""
        # Create conversation
        conversation = Conversation(
            id="api-export-md-conv",
            user_id=test_user.id,
            title="API Markdown Export",
        )
        db_session.add(conversation)
        await db_session.flush()

        msg = Message(
            conversation_id="api-export-md-conv",
            role="user",
            content="Test message for API export",
        )
        db_session.add(msg)
        await db_session.commit()

        headers = {"Authorization": f"Bearer {authenticated_user}"}
        response = client.get(
            "/api/export/conversations/api-export-md-conv/markdown",
            headers=headers
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        content = response.text
        assert "# API Markdown Export" in content

    @pytest.mark.asyncio
    async def test_export_conversation_json(self, client, authenticated_user, db_session, test_user):
        """Test exporting conversation to JSON via API."""
        # Create conversation
        conversation = Conversation(
            id="api-export-json-conv",
            user_id=test_user.id,
            title="API JSON Export",
        )
        db_session.add(conversation)
        await db_session.flush()

        msg = Message(
            conversation_id="api-export-json-conv",
            role="user",
            content="Test message for JSON export",
        )
        db_session.add(msg)
        await db_session.commit()

        headers = {"Authorization": f"Bearer {authenticated_user}"}
        response = client.get(
            "/api/export/conversations/api-export-json-conv/json",
            headers=headers
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert data["title"] == "API JSON Export"
        assert data["id"] == "api-export-json-conv"

    @pytest.mark.asyncio
    async def test_export_all_conversations_markdown(self, client, authenticated_user, db_session, test_user):
        """Test exporting all conversations to markdown."""
        # Create conversations
        for i in range(2):
            conv = Conversation(
                id=f"api-export-all-{i}",
                user_id=test_user.id,
                title=f"API All Export {i}",
            )
            db_session.add(conv)
        await db_session.commit()

        headers = {"Authorization": f"Bearer {authenticated_user}"}
        response = client.get("/api/export/all/markdown", headers=headers)

        assert response.status_code == 200
        content = response.text
        assert "# LLM Council Conversations Export" in content

    @pytest.mark.asyncio
    async def test_export_conversation_not_found(self, client, authenticated_user):
        """Test exporting non-existent conversation."""
        headers = {"Authorization": f"Bearer {authenticated_user}"}
        response = client.get(
            "/api/export/conversations/non-existent/markdown",
            headers=headers
        )

        # Should return 404 for non-existent conversation
        assert response.status_code == 404


class TestGlobalExportService:
    """Tests for global export service functions."""

    def test_get_export_service_singleton(self):
        """Test that get_export_service returns a singleton."""
        reset_export_service()

        service1 = get_export_service()
        service2 = get_export_service()

        assert service1 is service2

        reset_export_service()

    def test_reset_export_service(self):
        """Test resetting the global service."""
        service1 = get_export_service()

        reset_export_service()

        service2 = get_export_service()

        # Should be a new instance
        assert service1 is not service2

        reset_export_service()
