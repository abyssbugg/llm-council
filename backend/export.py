"""Export service for converting conversations to various formats.

This module provides functionality to:
- Export conversations to Markdown format
- Export conversations to PDF format
- Include metadata and formatting in exports
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from backend.models import User, Conversation, Message

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting conversations to different formats."""

    async def export_conversation_to_markdown(
        self,
        conversation_id: str,
        user_id: int,
        db: AsyncSession,
        include_metadata: bool = True,
    ) -> str:
        """Export a conversation to Markdown format.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID for ownership verification.
            db: Database session.
            include_metadata: Whether to include council metadata.

        Returns:
            Markdown formatted string.

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

        # Build markdown content
        lines = []
        lines.append(f"# {conversation.title}")
        lines.append("")
        lines.append(f"**Created:** {conversation.created_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Last Updated:** {conversation.updated_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Process messages
        for msg in conversation.messages:
            if msg.role == "user":
                lines.append(f"## 👤 User")
            else:
                lines.append(f"## 🤖 Assistant")

            lines.append("")
            lines.append(msg.content)
            lines.append("")

            # Add council metadata for assistant messages
            if msg.role == "assistant" and include_metadata and msg.stage1:
                lines.append("**Council Details:**")
                lines.append("")

                # Stage 1: Individual responses
                if msg.stage1 and "responses" in msg.stage1:
                    lines.append("### Stage 1: Individual Council Responses")
                    lines.append("")
                    for model_id, response in msg.stage1["responses"].items():
                        content_preview = response.get("content", "")[:100]
                        lines.append(f"- **{model_id}**: `{content_preview}...`")
                    lines.append("")

                # Stage 2: Rankings
                if msg.stage2 and "rankings" in msg.stage2:
                    lines.append("### Stage 2: Peer Rankings")
                    lines.append("")
                    for ranking in msg.stage2["rankings"]:
                        model = ranking.get("model", "Unknown")
                        lines.append(f"- **{model}** provided ranking")
                    lines.append("")

                # Aggregate rankings
                if msg.extra_metadata and "aggregate_rankings" in msg.extra_metadata:
                    lines.append("### Aggregate Rankings")
                    lines.append("")
                    for item in msg.extra_metadata["aggregate_rankings"]:
                        model = item.get("model", "Unknown")
                        avg_pos = item.get("average_position", 0)
                        lines.append(f"- **{model}**: Average position {avg_pos:.1f}")
                    lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    async def export_conversation_to_dict(
        self,
        conversation_id: str,
        user_id: int,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Export a conversation to a structured dictionary.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID for ownership verification.
            db: Database session.

        Returns:
            Dictionary with conversation data.

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

        # Build export data
        messages = []
        for msg in conversation.messages:
            message_data = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
            }

            if msg.role == "assistant":
                message_data["stage1"] = msg.stage1
                message_data["stage2"] = msg.stage2
                message_data["stage3"] = msg.stage3
                message_data["metadata"] = msg.extra_metadata

            messages.append(message_data)

        return {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "messages": messages,
        }

    async def export_multiple_conversations_to_markdown(
        self,
        conversation_ids: List[str],
        user_id: int,
        db: AsyncSession,
    ) -> str:
        """Export multiple conversations to a single Markdown file.

        Args:
            conversation_ids: List of conversation IDs.
            user_id: User ID for ownership verification.
            db: Database session.

        Returns:
            Markdown formatted string with all conversations.
        """
        lines = []
        lines.append("# LLM Council Conversations Export")
        lines.append("")
        lines.append(f"**Export Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("")
        lines.append("---")
        lines.append("")

        for conv_id in conversation_ids:
            try:
                conv_markdown = await self.export_conversation_to_markdown(
                    conv_id, user_id, db, include_metadata=True
                )
                lines.append(conv_markdown)
                lines.append("")
                lines.append("===")
                lines.append("")

            except ValueError as e:
                logger.warning(f"Skipping conversation {conv_id}: {e}")
                lines.append(f"*Conversation {conv_id} could not be exported.*")
                lines.append("")
                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    async def get_export_summary(
        self,
        user_id: int,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Get a summary of user's conversations for export.

        Args:
            user_id: User ID.
            db: Database session.

        Returns:
            Dictionary with export summary information.
        """
        result = await db.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
        )
        conversations = result.scalars().all()

        # Count messages
        total_messages = 0
        for conv in conversations:
            result = await db.execute(
                select(func.count(Message.id))
                .where(Message.conversation_id == conv.id)
            )
            total_messages += result.scalar_one() or 0

        return {
            "total_conversations": len(conversations),
            "total_messages": total_messages,
            "conversations": [
                {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat(),
                    "message_count": 0,  # Would require separate query
                }
                for conv in conversations
            ],
        }


# Singleton service instance
_export_service: Optional[ExportService] = None


def get_export_service() -> ExportService:
    """Get the global export service instance.

    Returns:
        The global ExportService instance.
    """
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
        logger.info("Initialized global export service")
    return _export_service


def reset_export_service() -> None:
    """Reset the global export service (useful for testing)."""
    global _export_service
    _export_service = None
    logger.info("Reset global export service")
