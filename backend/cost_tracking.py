"""Cost tracking and budget management service.

This module provides functionality to:
- Track API costs per user and conversation
- Enforce daily budget limits
- Calculate spending statistics
- Generate cost reports
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from fastapi import HTTPException, status

from backend.models import Cost, User
from backend.database import get_db

logger = logging.getLogger(__name__)


class CostTrackingService:
    """Service for tracking LLM API costs and managing budgets."""

    async def record_cost(
        self,
        user_id: int,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        conversation_id: Optional[str] = None,
        db: AsyncSession = None,
    ) -> Cost:
        """Record a cost entry after an LLM API call.

        Args:
            user_id: User ID who made the request.
            provider: Provider name (e.g., 'openrouter', 'chutes').
            model: Model identifier.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            cost_usd: Cost in USD.
            conversation_id: Optional conversation ID.
            db: Database session.

        Returns:
            Created Cost entry.
        """
        if db is None:
            async for db in get_db():
                return await self.record_cost(
                    user_id, provider, model, prompt_tokens,
                    completion_tokens, cost_usd, conversation_id, db
                )

        cost = Cost(
            user_id=user_id,
            conversation_id=conversation_id,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
        )

        db.add(cost)
        await db.commit()
        await db.refresh(cost)

        logger.info(
            f"Recorded cost: ${cost_usd:.4f} for user {user_id} "
            f"({prompt_tokens}+{completion_tokens} tokens, {provider}/{model})"
        )

        return cost

    async def get_daily_spending(
        self,
        user_id: int,
        db: AsyncSession,
    ) -> float:
        """Get total spending for a user today.

        Args:
            user_id: User ID.
            db: Database session.

        Returns:
            Total spending in USD for the current day.
        """
        # Start of today in UTC
        today_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        result = await db.execute(
            select(func.sum(Cost.cost_usd))
            .where(Cost.user_id == user_id)
            .where(Cost.created_at >= today_start)
        )
        total = result.scalar_one() or 0.0

        return float(total)

    async def check_budget_limit(
        self,
        user_id: int,
        db: AsyncSession,
        estimated_cost: float = 0.0,
    ) -> tuple[bool, float, float]:
        """Check if a user is within their daily budget limit.

        Args:
            user_id: User ID.
            db: Database session.
            estimated_cost: Estimated cost for the upcoming request.

        Returns:
            Tuple of (within_budget, current_spending, budget_limit).
            If within_budget is False, the user has exceeded their limit.
        """
        # Get user's budget limit
        result = await db.execute(
            select(User.daily_budget_limit)
            .where(User.id == user_id)
        )
        budget_limit = result.scalar_one_or_none()

        if budget_limit is None:
            budget_limit = 10.0  # Default $10/day

        # Get current spending
        current_spending = await self.get_daily_spending(user_id, db)

        # Check if adding estimated cost would exceed budget
        within_budget = (current_spending + estimated_cost) <= budget_limit

        if not within_budget:
            logger.warning(
                f"User {user_id} exceeded budget: "
                f"${current_spending:.2f} + ${estimated_cost:.2f} > ${budget_limit:.2f}"
            )

        return within_budget, current_spending, budget_limit

    async def get_cost_statistics(
        self,
        user_id: int,
        db: AsyncSession,
        days: int = 30,
    ) -> Dict:
        """Get cost statistics for a user over a period.

        Args:
            user_id: User ID.
            db: Database session.
            days: Number of days to look back (default: 30).

        Returns:
            Dictionary with cost statistics.
        """
        # Start of the period
        period_start = datetime.utcnow() - timedelta(days=days)

        # Total costs
        result = await db.execute(
            select(
                func.sum(Cost.cost_usd).label("total_cost"),
                func.sum(Cost.prompt_tokens).label("total_prompt_tokens"),
                func.sum(Cost.completion_tokens).label("total_completion_tokens"),
                func.count(Cost.id).label("request_count"),
            )
            .where(Cost.user_id == user_id)
            .where(Cost.created_at >= period_start)
        )
        row = result.one_or_none()

        stats = {
            "period_days": days,
            "total_cost": float(row.total_cost) if row.total_cost else 0.0,
            "total_prompt_tokens": int(row.total_prompt_tokens) if row.total_prompt_tokens else 0,
            "total_completion_tokens": int(row.total_completion_tokens) if row.total_completion_tokens else 0,
            "total_tokens": (int(row.total_prompt_tokens or 0) + int(row.total_completion_tokens or 0)),
            "request_count": int(row.request_count) if row.request_count else 0,
        }

        # Daily breakdown
        result = await db.execute(
            select(
                func.date(Cost.created_at).label("date"),
                func.sum(Cost.cost_usd).label("daily_cost"),
                func.count(Cost.id).label("daily_requests"),
            )
            .where(Cost.user_id == user_id)
            .where(Cost.created_at >= period_start)
            .group_by(func.date(Cost.created_at))
            .order_by(func.date(Cost.created_at).desc())
        )
        daily_breakdown = [
            {
                "date": str(row.date),
                "cost": float(row.daily_cost),
                "requests": int(row.daily_requests),
            }
            for row in result
        ]

        stats["daily_breakdown"] = daily_breakdown

        # By provider
        result = await db.execute(
            select(
                Cost.provider,
                func.sum(Cost.cost_usd).label("provider_cost"),
                func.count(Cost.id).label("provider_requests"),
            )
            .where(Cost.user_id == user_id)
            .where(Cost.created_at >= period_start)
            .group_by(Cost.provider)
        )
        by_provider = [
            {
                "provider": row.provider,
                "cost": float(row.provider_cost),
                "requests": int(row.provider_requests),
            }
            for row in result
        ]

        stats["by_provider"] = by_provider

        # By model
        result = await db.execute(
            select(
                Cost.model,
                func.sum(Cost.cost_usd).label("model_cost"),
                func.count(Cost.id).label("model_requests"),
            )
            .where(Cost.user_id == user_id)
            .where(Cost.created_at >= period_start)
            .group_by(Cost.model)
            .order_by(func.sum(Cost.cost_usd).desc())
            .limit(10)
        )
        by_model = [
            {
                "model": row.model,
                "cost": float(row.model_cost),
                "requests": int(row.model_requests),
            }
            for row in result
        ]

        stats["by_model"] = by_model

        return stats

    async def get_conversation_costs(
        self,
        conversation_id: str,
        db: AsyncSession,
    ) -> List[Cost]:
        """Get all costs for a conversation.

        Args:
            conversation_id: Conversation ID.
            db: Database session.

        Returns:
            List of Cost entries for the conversation.
        """
        result = await db.execute(
            select(Cost)
            .where(Cost.conversation_id == conversation_id)
            .order_by(Cost.created_at)
        )
        return list(result.scalars().all())

    async def get_total_conversation_cost(
        self,
        conversation_id: str,
        db: AsyncSession,
    ) -> float:
        """Get total cost for a conversation.

        Args:
            conversation_id: Conversation ID.
            db: Database session.

        Returns:
            Total cost in USD.
        """
        result = await db.execute(
            select(func.sum(Cost.cost_usd))
            .where(Cost.conversation_id == conversation_id)
        )
        total = result.scalar_one() or 0.0
        return float(total)


# Singleton service instance
_cost_service: Optional[CostTrackingService] = None


def get_cost_service() -> CostTrackingService:
    """Get the global cost tracking service instance.

    Returns:
        The global CostTrackingService instance.
    """
    global _cost_service
    if _cost_service is None:
        _cost_service = CostTrackingService()
        logger.info("Initialized global cost tracking service")
    return _cost_service


def reset_cost_service() -> None:
    """Reset the global cost service (useful for testing)."""
    global _cost_service
    _cost_service = None
    logger.info("Reset global cost service")
