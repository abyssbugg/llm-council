"""Authentication service for user management."""

import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status

from ..database import get_db
from ..models import User, UserConfig
from . import (
    verify_password,
    get_password_hash,
    create_access_token,
    validate_password_strength,
    sanitize_string,
    sanitize_email,
    UserCreate,
    UserLogin,
    UserResponse,
    Token,
)

logger = logging.getLogger(__name__)


class AuthService:
    """Service for handling authentication operations."""
    
    async def register_user(
        self,
        user_data: UserCreate,
        db: AsyncSession,
    ) -> UserResponse:
        """Register a new user.
        
        Args:
            user_data: User registration data.
            db: Database session.
        
        Returns:
            Created user data.
        
        Raises:
            HTTPException: 400 if validation fails, 409 if user exists.
        """
        # Sanitize inputs
        email = sanitize_email(user_data.email)
        password = user_data.password
        display_name = sanitize_string(user_data.display_name or "") if user_data.display_name else None
        
        # Validate email format
        if "@" not in email or "." not in email.split("@")[-1]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email address"
            )
        
        # Check if user already exists
        result = await db.execute(select(User).where(User.email == email))
        if result.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists"
            )
        
        # Validate password strength
        is_valid, errors = validate_password_strength(password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"password_errors": errors}
            )
        
        # Hash password
        hashed_password = get_password_hash(password)
        
        # Create user
        user = User(
            email=email,
            hashed_password=hashed_password,
            display_name=display_name,
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        # Create default user config
        user_config = UserConfig(
            user_id=user.id,
            council_models=[],
            chairman_model="google/gemini-pro",
            temperature=0.7,
            max_tokens=4096,
        )
        db.add(user_config)
        await db.commit()
        
        logger.info(f"Registered new user: {user.id}")
        
        return UserResponse(
            id=str(user.id),
            email=user.email,
            display_name=user.display_name,
            daily_budget_limit=user.daily_budget_limit,
            is_active=user.is_active,
        )
    
    async def authenticate_user(
        self,
        login_data: UserLogin,
        db: AsyncSession,
    ) -> Token:
        """Authenticate a user and return an access token.
        
        Args:
            login_data: User login credentials.
            db: Database session.
        
        Returns:
            JWT access token.
        
        Raises:
            HTTPException: 401 if credentials are invalid.
        """
        # Sanitize inputs
        email = sanitize_email(login_data.email)
        password = login_data.password
        
        # Find user
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if user is None:
            # Use generic error message for security
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )
        
        # Verify password
        if not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Create access token
        access_token = create_access_token(
            data={"sub": user.email, "user_id": user.id}
        )
        
        logger.info(f"User authenticated: {user.id}")
        
        return Token(access_token=access_token)
    
    async def get_user_by_id(
        self,
        user_id: int,
        db: AsyncSession,
    ) -> UserResponse:
        """Get a user by ID.
        
        Args:
            user_id: User ID to fetch.
            db: Database session.
        
        Returns:
            User data.
        
        Raises:
            HTTPException: 404 if user not found.
        """
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=str(user.id),
            email=user.email,
            display_name=user.display_name,
            daily_budget_limit=user.daily_budget_limit,
            is_active=user.is_active,
        )
    
    async def update_user(
        self,
        user_id: int,
        display_name: Optional[str] = None,
        daily_budget_limit: Optional[float] = None,
        db: AsyncSession = None,
    ) -> UserResponse:
        """Update user information.
        
        Args:
            user_id: User ID to update.
            display_name: New display name.
            daily_budget_limit: New daily budget limit.
            db: Database session.
        
        Returns:
            Updated user data.
        
        Raises:
            HTTPException: 404 if user not found, 400 if invalid data.
        """
        if db is None:
            async for db in get_db():
                return await self.update_user(
                    user_id, display_name, daily_budget_limit, db
                )
        
        # Fetch user
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update fields
        if display_name is not None:
            user.display_name = sanitize_string(display_name, max_length=255)
        
        if daily_budget_limit is not None:
            if daily_budget_limit < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Budget limit must be non-negative"
                )
            user.daily_budget_limit = daily_budget_limit
        
        await db.commit()
        await db.refresh(user)
        
        logger.info(f"User updated: {user_id}")
        
        return UserResponse(
            id=str(user.id),
            email=user.email,
            display_name=user.display_name,
            daily_budget_limit=user.daily_budget_limit,
            is_active=user.is_active,
        )
    
    async def change_password(
        self,
        user_id: int,
        current_password: str,
        new_password: str,
        db: AsyncSession,
    ) -> None:
        """Change a user's password.
        
        Args:
            user_id: User ID.
            current_password: Current password for verification.
            new_password: New password to set.
            db: Database session.
        
        Raises:
            HTTPException: 400 if validation fails, 401 if current password is wrong.
        """
        # Fetch user
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        
        # Validate new password strength
        is_valid, errors = validate_password_strength(new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"password_errors": errors}
            )
        
        # Update password
        user.hashed_password = get_password_hash(new_password)
        
        await db.commit()
        
        logger.info(f"Password changed for user: {user_id}")


# Singleton service instance
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get the global auth service instance.
    
    Returns:
        The global AuthService instance.
    """
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
        logger.info("Initialized global auth service")
    return _auth_service


def reset_auth_service() -> None:
    """Reset the global auth service (useful for testing)."""
    global _auth_service
    _auth_service = None
    logger.info("Reset global auth service")
