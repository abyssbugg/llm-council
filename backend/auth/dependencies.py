"""FastAPI dependencies for authentication and database access."""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database import get_db
from ..models import User
from . import decode_access_token

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get the current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials containing the JWT token.
        db: Database session.
    
    Returns:
        The authenticated user.
    
    Raises:
        HTTPException: 401 if token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Decode token
    token_data = decode_access_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception
    
    # Check if user exists
    result = await db.execute(
        select(User).where(User.email == token_data.email)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get the current active user (alias for get_current_user).
    
    This function exists for clarity and future extensibility
    (e.g., adding additional checks like email verification).
    
    Args:
        current_user: The current authenticated user.
    
    Returns:
        The active user.
    """
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Get the current user if authenticated, None otherwise.

    This is useful for endpoints that work for both authenticated
    and anonymous users.

    Args:
        credentials: Optional HTTP Bearer credentials.
        db: Database session.

    Returns:
        The authenticated user, or None if not authenticated.
    """
    if credentials is None or credentials.credentials is None:
        return None

    token_data = decode_access_token(credentials.credentials)
    if token_data is None:
        return None

    result = await db.execute(
        select(User).where(User.email == token_data.email)
    )
    user = result.scalar_one_or_none()

    if user and user.is_active:
        return user

    return None
