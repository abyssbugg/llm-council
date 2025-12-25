"""Export API endpoints for conversation exports."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from pydantic import BaseModel

from backend.database import get_db
from backend.models import User, Conversation
from backend.export import get_export_service
from backend.auth.routes import get_current_user

router = APIRouter(prefix="/export", tags=["export"])


class ExportRequest(BaseModel):
    """Request model for export operations."""
    conversation_ids: List[str]
    format: str = "markdown"  # "markdown" or "json"
    include_metadata: bool = True


class ExportResponse(BaseModel):
    """Response model for export operations."""
    content: str
    format: str
    content_type: str
    filename: str


@router.get("/summary")
async def get_export_summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get summary of conversations available for export."""
    export_service = get_export_service()
    summary = await export_service.get_export_summary(current_user.id, db)
    return summary


@router.post("/conversations")
async def export_conversations(
    request: ExportRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export one or more conversations.

    Returns the exported content in the specified format.
    """
    export_service = get_export_service()

    if request.format == "markdown":
        if len(request.conversation_ids) == 1:
            # Single conversation export
            content = await export_service.export_conversation_to_markdown(
                request.conversation_ids[0],
                current_user.id,
                db,
                include_metadata=request.include_metadata,
            )
            # Get title for filename
            conv_result = await db.execute(
                select(Conversation).where(Conversation.id == request.conversation_ids[0])
            )
            conv = conv_result.scalar_one_or_none()
            title = conv.title if conv else "conversation"
            filename = f"{title}.md"

        else:
            # Multiple conversations export
            content = await export_service.export_multiple_conversations_to_markdown(
                request.conversation_ids,
                current_user.id,
                db,
            )
            filename = "llm_council_export.md"

        return PlainTextResponse(
            content=content,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    elif request.format == "json":
        if len(request.conversation_ids) != 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON export only supports single conversation",
            )

        # Export as JSON
        data = await export_service.export_conversation_to_dict(
            request.conversation_ids[0],
            current_user.id,
            db,
        )

        import json
        content = json.dumps(data, indent=2)

        # Get title for filename
        conv_result = await db.execute(
            select(Conversation).where(Conversation.id == request.conversation_ids[0])
        )
        conv = conv_result.scalar_one_or_none()
        title = conv.title if conv else "conversation"
        filename = f"{title}.json"

        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {request.format}. Supported: markdown, json",
        )


@router.get("/conversations/{conversation_id}/markdown")
async def export_conversation_markdown(
    conversation_id: str,
    include_metadata: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export a single conversation to Markdown format.

    Query parameters:
    - include_metadata: Whether to include council metadata (default: true)
    """
    export_service = get_export_service()

    try:
        content = await export_service.export_conversation_to_markdown(
            conversation_id,
            current_user.id,
            db,
            include_metadata=include_metadata,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Get title for filename
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = conv_result.scalar_one_or_none()
    title = conv.title if conv else "conversation"
    filename = f"{title}.md"

    return PlainTextResponse(
        content=content,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get("/conversations/{conversation_id}/json")
async def export_conversation_json(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export a single conversation to JSON format."""
    export_service = get_export_service()

    try:
        data = await export_service.export_conversation_to_dict(
            conversation_id,
            current_user.id,
            db,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    import json
    content = json.dumps(data, indent=2)

    # Get title for filename
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = conv_result.scalar_one_or_none()
    title = conv.title if conv else "conversation"
    filename = f"{title}.json"

    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get("/all/markdown")
async def export_all_conversations_markdown(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export all user conversations to a single Markdown file."""
    export_service = get_export_service()

    # Get all user's conversation IDs
    result = await db.execute(
        select(Conversation.id)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
    )
    conversation_ids = [row[0] for row in result]

    content = await export_service.export_multiple_conversations_to_markdown(
        conversation_ids,
        current_user.id,
        db,
    )

    filename = "llm_council_all_conversations.md"

    return PlainTextResponse(
        content=content,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
