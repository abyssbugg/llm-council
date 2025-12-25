"""Tests for the cost tracking module."""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from backend.cost_tracking import (
    CostTrackingService,
    get_cost_service,
    reset_cost_service,
)
from backend.models import Cost, User, Base


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
def cost_service():
    """Fixture providing a test cost service."""
    reset_cost_service()
    return get_cost_service()


class TestCostTrackingService:
    """Tests for CostTrackingService."""

    @pytest.mark.asyncio
    async def test_record_cost(self, db_session, cost_service):
        """Test recording a cost entry."""
        # Create a test user
        user = User(
            email="test@example.com",
            hashed_password="hash",
            daily_budget_limit=10.0,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Record a cost
        cost = await cost_service.record_cost(
            user_id=user.id,
            provider="openrouter",
            model="openai/gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.005,
            conversation_id="conv-123",
            db=db_session,
        )

        assert cost.id is not None
        assert cost.user_id == user.id
        assert cost.provider == "openrouter"
        assert cost.model == "openai/gpt-4"
        assert cost.prompt_tokens == 100
        assert cost.completion_tokens == 50
        assert cost.total_tokens == 150
        assert cost.cost_usd == 0.005

    @pytest.mark.asyncio
    async def test_get_daily_spending(self, db_session, cost_service):
        """Test calculating daily spending."""
        # Create a test user
        user = User(
            email="test@example.com",
            hashed_password="hash",
            daily_budget_limit=10.0,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Record costs from today
        await cost_service.record_cost(
            user_id=user.id,
            provider="openrouter",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.005,
            db=db_session,
        )
        await cost_service.record_cost(
            user_id=user.id,
            provider="chutes",
            model="claude-3",
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=0.010,
            db=db_session,
        )

        # Get daily spending
        spending = await cost_service.get_daily_spending(user.id, db_session)

        assert spending == 0.015  # 0.005 + 0.010

    @pytest.mark.asyncio
    async def test_check_budget_limit_within(self, db_session, cost_service):
        """Test budget check when within limit."""
        # Create a test user
        user = User(
            email="test@example.com",
            hashed_password="hash",
            daily_budget_limit=10.0,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Record some costs
        await cost_service.record_cost(
            user_id=user.id,
            provider="openrouter",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=5.0,  # Half of budget
            db=db_session,
        )

        # Check budget
        within_budget, current, limit = await cost_service.check_budget_limit(
            user.id, db_session, estimated_cost=1.0
        )

        assert within_budget is True
        assert current == 5.0
        assert limit == 10.0

    @pytest.mark.asyncio
    async def test_check_budget_limit_exceeded(self, db_session, cost_service):
        """Test budget check when limit exceeded."""
        # Create a test user
        user = User(
            email="test@example.com",
            hashed_password="hash",
            daily_budget_limit=10.0,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Record costs up to limit
        await cost_service.record_cost(
            user_id=user.id,
            provider="openrouter",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=9.5,
            db=db_session,
        )

        # Check budget with a cost that would exceed limit
        within_budget, current, limit = await cost_service.check_budget_limit(
            user.id, db_session, estimated_cost=1.0  # Would make 10.5 > 10.0
        )

        assert within_budget is False
        assert current == 9.5
        assert limit == 10.0

    @pytest.mark.asyncio
    async def test_check_budget_limit_at_limit(self, db_session, cost_service):
        """Test budget check when exactly at limit."""
        # Create a test user
        user = User(
            email="test@example.com",
            hashed_password="hash",
            daily_budget_limit=10.0,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Record costs up to limit
        await cost_service.record_cost(
            user_id=user.id,
            provider="openrouter",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=10.0,
            db=db_session,
        )

        # Check budget with zero estimated cost
        within_budget, current, limit = await cost_service.check_budget_limit(
            user.id, db_session, estimated_cost=0.0
        )

        assert within_budget is True
        assert current == 10.0
        assert limit == 10.0

    @pytest.mark.asyncio
    async def test_get_cost_statistics(self, db_session, cost_service):
        """Test getting cost statistics."""
        # Create a test user
        user = User(
            email="test@example.com",
            hashed_password="hash",
            daily_budget_limit=10.0,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Record some costs
        await cost_service.record_cost(
            user_id=user.id,
            provider="openrouter",
            model="gpt-4",
            prompt_tokens=1000,
            completion_tokens=500,
            cost_usd=0.05,
            conversation_id="conv-1",
            db=db_session,
        )
        await cost_service.record_cost(
            user_id=user.id,
            provider="chutes",
            model="claude-3",
            prompt_tokens=500,
            completion_tokens=250,
            cost_usd=0.03,
            conversation_id="conv-1",
            db=db_session,
        )

        # Get statistics
        stats = await cost_service.get_cost_statistics(user.id, db_session, days=30)

        assert stats["period_days"] == 30
        assert stats["total_cost"] == 0.08
        assert stats["total_prompt_tokens"] == 1500
        assert stats["total_completion_tokens"] == 750
        assert stats["total_tokens"] == 2250
        assert stats["request_count"] == 2
        assert len(stats["by_provider"]) == 2
        assert len(stats["by_model"]) == 2
        assert len(stats["daily_breakdown"]) >= 1  # At least today

    @pytest.mark.asyncio
    async def test_get_conversation_costs(self, db_session, cost_service):
        """Test getting costs for a conversation."""
        # Create a test user
        user = User(
            email="test@example.com",
            hashed_password="hash",
            daily_budget_limit=10.0,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        conversation_id = "conv-test-123"

        # Record costs for a conversation
        await cost_service.record_cost(
            user_id=user.id,
            provider="openrouter",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.01,
            conversation_id=conversation_id,
            db=db_session,
        )
        await cost_service.record_cost(
            user_id=user.id,
            provider="chutes",
            model="claude-3",
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=0.02,
            conversation_id=conversation_id,
            db=db_session,
        )

        # Get conversation costs
        costs = await cost_service.get_conversation_costs(conversation_id, db_session)

        assert len(costs) == 2
        assert all(c.conversation_id == conversation_id for c in costs)

    @pytest.mark.asyncio
    async def test_get_total_conversation_cost(self, db_session, cost_service):
        """Test getting total cost for a conversation."""
        # Create a test user
        user = User(
            email="test@example.com",
            hashed_password="hash",
            daily_budget_limit=10.0,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        conversation_id = "conv-cost-test"

        # Record costs for a conversation
        await cost_service.record_cost(
            user_id=user.id,
            provider="openrouter",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.01,
            conversation_id=conversation_id,
            db=db_session,
        )
        await cost_service.record_cost(
            user_id=user.id,
            provider="chutes",
            model="claude-3",
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=0.02,
            conversation_id=conversation_id,
            db=db_session,
        )

        # Get total cost
        total = await cost_service.get_total_conversation_cost(conversation_id, db_session)

        assert total == 0.03  # 0.01 + 0.02

    @pytest.mark.asyncio
    async def test_get_conversation_costs_empty(self, db_session, cost_service):
        """Test getting costs for a conversation with no costs."""
        costs = await cost_service.get_conversation_costs("nonexistent-conv", db_session)
        assert costs == []

    @pytest.mark.asyncio
    async def test_get_total_conversation_cost_empty(self, db_session, cost_service):
        """Test getting total cost for a conversation with no costs."""
        total = await cost_service.get_total_conversation_cost("nonexistent-conv", db_session)
        assert total == 0.0


class TestGlobalCostService:
    """Tests for global cost service functions."""

    def test_get_cost_service_singleton(self):
        """Test that get_cost_service returns a singleton."""
        reset_cost_service()

        service1 = get_cost_service()
        service2 = get_cost_service()

        assert service1 is service2

        reset_cost_service()

    def test_reset_cost_service(self):
        """Test resetting the global service."""
        service1 = get_cost_service()

        reset_cost_service()

        service2 = get_cost_service()
        # Should be a new instance
        assert service1 is not service2

        reset_cost_service()
