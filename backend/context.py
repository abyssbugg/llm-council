"""Context management service for conversation history and memory.

This module provides functionality to:
- Manage conversation context windows
- Build conversation history for LLM calls
- Implement context management strategies (sliding window, summary, etc.)
"""

import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.models import User, UserConfig, Conversation, Message

logger = logging.getLogger(__name__)


class ContextManagementService:
    """Service for managing conversation context and history."""

    async def get_conversation_history(
        self,
        conversation_id: str,
        user_id: int,
        max_messages: Optional[int] = None,
        db: AsyncSession = None,
    ) -> List[Dict[str, str]]:
        """Get conversation history for context.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID for ownership verification.
            max_messages: Maximum messages to include (None for user default).
            db: Database session.

        Returns:
            List of message dictionaries with 'role' and 'content' keys.

        Raises:
            ValueError: If conversation doesn't exist or user doesn't own it.
        """
        # Get user's max_history_messages preference if not specified
        if max_messages is None:
            result = await db.execute(
                select(UserConfig).where(UserConfig.user_id == user_id)
            )
            user_config = result.scalar_one_or_none()
            max_messages = user_config.max_history_messages if user_config else 10

        # Verify conversation ownership and get messages
        result = await db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == user_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise ValueError(
                f"Conversation {conversation_id} not found or access denied"
            )

        # Build history from messages
        # Get recent messages up to max_messages (most recent last)
        messages = conversation.messages
        if len(messages) > max_messages:
            messages = messages[-max_messages:]

        history = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        logger.info(
            f"Retrieved {len(history)} messages for conversation {conversation_id} "
            f"(max: {max_messages})"
        )

        return history

    async def build_context_for_generation(
        self,
        conversation_id: str,
        user_id: int,
        current_message: str,
        include_system_prompt: bool = True,
        db: AsyncSession = None,
    ) -> tuple[List[Dict[str, str]], Optional[str]]:
        """Build full context for LLM generation.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID.
            current_message: Current user message to add.
            include_system_prompt: Whether to include user's system prompt.
            db: Database session.

        Returns:
            Tuple of (messages_list, system_prompt).
            messages_list includes history + current message in LLM format.
        """
        # Get user config for system prompt and context settings
        result = await db.execute(
            select(UserConfig).where(UserConfig.user_id == user_id)
        )
        user_config = result.scalar_one_or_none()

        system_prompt = None
        max_messages = 10

        if user_config:
            system_prompt = user_config.system_prompt
            max_messages = user_config.max_history_messages

        # Get conversation history
        history = await self.get_conversation_history(
            conversation_id,
            user_id,
            max_messages=max_messages,
            db=db,
        )

        # Build messages list
        messages = []

        # Add system prompt if provided
        if include_system_prompt and system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add history (excluding current message which will be added by caller)
        messages.extend(history)

        # Note: Current message is added by the caller
        # This allows the caller to control when to add it

        return messages, system_prompt

    async def get_context_window_info(
        self,
        conversation_id: str,
        user_id: int,
        db: AsyncSession = None,
    ) -> Dict[str, Any]:
        """Get information about the current context window.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID.
            db: Database session.

        Returns:
            Dictionary with context window information including:
            - total_messages: Total messages in conversation
            - context_messages: Messages included in context window
            - max_messages: User's max_history_messages setting
            - estimated_tokens: Estimated token count for context
        """
        # Get user config
        result = await db.execute(
            select(UserConfig).where(UserConfig.user_id == user_id)
        )
        user_config = result.scalar_one_or_none()

        max_messages = user_config.max_history_messages if user_config else 10

        # Get conversation and messages
        result = await db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == user_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise ValueError(
                f"Conversation {conversation_id} not found or access denied"
            )

        total_messages = len(conversation.messages)
        context_messages = min(total_messages, max_messages)

        # Rough token estimation (4 chars per token average)
        messages_in_context = conversation.messages[-context_messages:]
        total_chars = sum(len(msg.content) for msg in messages_in_context)
        estimated_tokens = total_chars // 4

        return {
            "total_messages": total_messages,
            "context_messages": context_messages,
            "max_messages": max_messages,
            "estimated_tokens": estimated_tokens,
            "truncated_messages": total_messages - context_messages if total_messages > max_messages else 0,
        }

    async def truncate_conversation(
        self,
        conversation_id: str,
        user_id: int,
        keep_messages: int,
        db: AsyncSession = None,
    ) -> int:
        """Truncate a conversation to keep only the most recent messages.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID.
            keep_messages: Number of most recent messages to keep.
            db: Database session.

        Returns:
            Number of messages deleted.

        Raises:
            ValueError: If conversation doesn't exist or user doesn't own it.
        """
        # Get conversation with messages
        result = await db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == user_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise ValueError(
                f"Conversation {conversation_id} not found or access denied"
            )

        total_messages = len(conversation.messages)

        if total_messages <= keep_messages:
            return 0  # Nothing to truncate

        # Delete older messages (keep the most recent ones)
        messages_to_delete = conversation.messages[:-keep_messages]

        for msg in messages_to_delete:
            await db.delete(msg)

        await db.commit()

        deleted_count = len(messages_to_delete)
        logger.info(
            f"Truncated conversation {conversation_id}: "
            f"deleted {deleted_count} old messages, kept {keep_messages}"
        )

        return deleted_count


# Singleton service instance
_context_service: Optional[ContextManagementService] = None


def get_context_service() -> ContextManagementService:
    """Get the global context management service instance.

    Returns:
        The global ContextManagementService instance.
    """
    global _context_service
    if _context_service is None:
        _context_service = ContextManagementService()
        logger.info("Initialized global context management service")
    return _context_service


def reset_context_service() -> None:
    """Reset the global context service (useful for testing)."""
    global _context_service
    _context_service = None
    logger.info("Reset global context service")
