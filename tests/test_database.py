"""Tests for database models and relationships."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from backend.database import Base, get_db
from backend.models import User, Conversation, Message, Cost, UserConfig


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


class TestUser:
    """Tests for User model."""

    @pytest.mark.asyncio
    async def test_create_user(self, db_session):
        """Test creating a user with required fields."""
        user = User(
            email="test@example.com",
            hashed_password="hashed_password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.hashed_password == "hashed_password123"
        assert user.is_active is True
        assert user.daily_budget_limit == 10.0

    @pytest.mark.asyncio
    async def test_user_with_optional_fields(self, db_session):
        """Test creating a user with optional fields."""
        user = User(
            email="user@example.com",
            hashed_password="password123",
            display_name="Test User",
            daily_budget_limit=50.0
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        assert user.display_name == "Test User"
        assert user.daily_budget_limit == 50.0

    @pytest.mark.asyncio
    async def test_user_unique_email(self, db_session):
        """Test that email must be unique."""
        user1 = User(
            email="duplicate@example.com",
            hashed_password="password1"
        )
        user2 = User(
            email="duplicate@example.com",
            hashed_password="password2"
        )
        db_session.add(user1)
        await db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(Exception):  # IntegrityError
            await db_session.commit()


class TestConversation:
    """Tests for Conversation model."""

    @pytest.mark.asyncio
    async def test_create_conversation(self, db_session):
        """Test creating a conversation for a user."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        conversation = Conversation(
            id="test-conv-123",
            user_id=user.id,
            title="Test Conversation"
        )
        db_session.add(conversation)
        await db_session.commit()
        await db_session.refresh(conversation)
        
        assert conversation.id == "test-conv-123"
        assert conversation.user_id == user.id
        assert conversation.title == "Test Conversation"
        assert conversation.created_at is not None

    @pytest.mark.asyncio
    async def test_conversation_default_title(self, db_session):
        """Test that conversation has default title."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        conversation = Conversation(
            id="test-conv-456",
            user_id=user.id
        )
        db_session.add(conversation)
        await db_session.commit()
        await db_session.refresh(conversation)
        
        assert conversation.title == "New Conversation"

    @pytest.mark.asyncio
    async def test_user_conversation_relationship(self, db_session):
        """Test the bidirectional user-conversation relationship."""
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        
        conv1 = Conversation(id="conv-1", user_id=user.id)
        conv2 = Conversation(id="conv-2", user_id=user.id)
        db_session.add_all([conv1, conv2])
        await db_session.commit()
        
        # Query with eager loading using selectinload
        result = await db_session.execute(
            select(User)
            .options(selectinload(User.conversations))
            .where(User.id == user.id)
        )
        user = result.scalar_one()
        
        assert len(user.conversations) == 2
        assert user.conversations[0].id in ["conv-1", "conv-2"]


class TestMessage:
    """Tests for Message model."""

    @pytest.mark.asyncio
    async def test_create_user_message(self, db_session):
        """Test creating a user message."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        conversation = Conversation(
            id="test-conv-789",
            user_id=user.id
        )
        db_session.add(conversation)
        await db_session.commit()
        await db_session.refresh(conversation)
        
        message = Message(
            conversation_id=conversation.id,
            role="user",
            content="Hello, world!"
        )
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        
        assert message.id is not None
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.conversation_id == conversation.id

    @pytest.mark.asyncio
    async def test_create_assistant_message_with_stages(self, db_session):
        """Test creating an assistant message with council stages."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        conversation = Conversation(
            id="test-conv-999",
            user_id=user.id
        )
        db_session.add(conversation)
        await db_session.commit()
        await db_session.refresh(conversation)
        
        stage1_data = {"response_a": "Answer from model A"}
        stage2_data = {"rankings": [{"model": "A", "rank": 1}]}
        stage3_data = {"final_answer": "Synthesized response"}
        
        message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content="Final answer",
            stage1=stage1_data,
            stage2=stage2_data,
            stage3=stage3_data,
            extra_metadata={"label_to_model": {"A": "openai/gpt-4"}}
        )
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        
        assert message.role == "assistant"
        assert message.stage1 == stage1_data
        assert message.stage2 == stage2_data
        assert message.stage3 == stage3_data
        assert message.extra_metadata == {"label_to_model": {"A": "openai/gpt-4"}}

    @pytest.mark.asyncio
    async def test_conversation_message_relationship(self, db_session):
        """Test the conversation-message relationship."""
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        
        conversation = Conversation(
            id="test-conv-rel",
            user_id=user.id
        )
        db_session.add(conversation)
        await db_session.commit()
        
        msg1 = Message(conversation_id=conversation.id, role="user", content="Hi")
        msg2 = Message(conversation_id=conversation.id, role="assistant", content="Hello")
        db_session.add_all([msg1, msg2])
        await db_session.commit()
        
        # Query with eager loading using selectinload
        result = await db_session.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation.id)
        )
        conversation = result.scalar_one()
        
        assert len(conversation.messages) == 2
        assert conversation.messages[0].content in ["Hi", "Hello"]


class TestCost:
    """Tests for Cost model."""

    @pytest.mark.asyncio
    async def test_create_cost_record(self, db_session):
        """Test creating a cost tracking record."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        cost = Cost(
            user_id=user.id,
            provider="openrouter",
            model="openai/gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.003
        )
        db_session.add(cost)
        await db_session.commit()
        await db_session.refresh(cost)
        
        assert cost.id is not None
        assert cost.provider == "openrouter"
        assert cost.model == "openai/gpt-4"
        assert cost.total_tokens == 150
        assert cost.cost_usd == 0.003

    @pytest.mark.asyncio
    async def test_cost_with_conversation(self, db_session):
        """Test cost linked to a conversation."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        conversation = Conversation(
            id="cost-conv-123",
            user_id=user.id
        )
        db_session.add(conversation)
        await db_session.commit()
        await db_session.refresh(conversation)
        
        cost = Cost(
            user_id=user.id,
            conversation_id=conversation.id,
            provider="openrouter",
            model="anthropic/claude-3",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            cost_usd=0.006
        )
        db_session.add(cost)
        await db_session.commit()
        await db_session.refresh(cost)
        
        assert cost.conversation_id == conversation.id


class TestUserConfig:
    """Tests for UserConfig model."""

    @pytest.mark.asyncio
    async def test_create_user_config(self, db_session):
        """Test creating user configuration."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        config = UserConfig(
            user_id=user.id,
            council_models=["openai/gpt-4", "anthropic/claude-3"],
            chairman_model="google/gemini-pro",
            temperature=0.8,
            max_tokens=2048,
            system_prompt="You are a helpful assistant."
        )
        db_session.add(config)
        await db_session.commit()
        await db_session.refresh(config)
        
        assert config.user_id == user.id
        assert config.council_models == ["openai/gpt-4", "anthropic/claude-3"]
        assert config.chairman_model == "google/gemini-pro"
        assert config.temperature == 0.8
        assert config.max_tokens == 2048

    @pytest.mark.asyncio
    async def test_user_config_defaults(self, db_session):
        """Test default values for user config."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        config = UserConfig(user_id=user.id)
        db_session.add(config)
        await db_session.commit()
        await db_session.refresh(config)
        
        assert config.council_models == []
        assert config.chairman_model == "google/gemini-pro"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.max_history_messages == 10

    @pytest.mark.asyncio
    async def test_user_config_unique_per_user(self, db_session):
        """Test that each user can have only one config."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        config1 = UserConfig(user_id=user.id)
        db_session.add(config1)
        await db_session.commit()
        
        config2 = UserConfig(user_id=user.id)
        db_session.add(config2)
        with pytest.raises(Exception):  # IntegrityError for unique constraint
            await db_session.commit()


class TestCascadeDelete:
    """Tests for cascade delete behavior."""

    @pytest.mark.asyncio
    async def test_delete_user_cascades_to_conversations(self, db_session):
        """Test that deleting a user deletes their conversations."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        conversation = Conversation(
            id="cascade-conv-123",
            user_id=user.id
        )
        db_session.add(conversation)
        await db_session.commit()
        
        await db_session.delete(user)
        await db_session.commit()
        
        # Verify conversation is deleted
        from sqlalchemy import select
        result = await db_session.execute(
            select(Conversation).where(Conversation.id == "cascade-conv-123")
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_delete_conversation_cascades_to_messages(self, db_session):
        """Test that deleting a conversation deletes its messages."""
        user = User(
            email="test@example.com",
            hashed_password="password123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        conversation = Conversation(
            id="cascade-msg-conv",
            user_id=user.id
        )
        db_session.add(conversation)
        await db_session.commit()
        await db_session.refresh(conversation)
        
        message = Message(
            conversation_id=conversation.id,
            role="user",
            content="Test message"
        )
        db_session.add(message)
        await db_session.commit()
        
        await db_session.delete(conversation)
        await db_session.commit()
        
        # Verify message is deleted
        from sqlalchemy import select
        result = await db_session.execute(
            select(Message).where(Message.conversation_id == conversation.id)
        )
        assert result.scalars().all() == []
