"""Authentication API routes.

This module provides FastAPI routes for user authentication:
- POST /api/auth/register - User registration
- POST /api/auth/login - User login with JWT token
- GET /api/auth/me - Get current user info
- POST /api/auth/change-password - Change password
- PUT /api/auth/me - Update user info
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict

from . import (
    Token,
    UserCreate,
    UserLogin,
    UserResponse,
    UserChangePassword,
    UserUpdate,
)
from .dependencies import get_current_user, get_current_active_user
from .service import AuthService, get_auth_service
from ..database import get_db
from ..models import User


router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user account.

    Validates email format, password strength, and checks for duplicates.
    Creates a default UserConfig for the new user.

    Args:
        user_data: User registration data (email, password, optional display_name)
        db: Database session

    Returns:
        UserResponse with created user data

    Raises:
        HTTPException 400: If email invalid, password weak, or user exists
    """
    auth_service = get_auth_service()

    try:
        user = await auth_service.register_user(user_data, db)
        return UserResponse(
            id=str(user.id),
            email=user.email,
            display_name=user.display_name,
            is_active=user.is_active,
            daily_budget_limit=user.daily_budget_limit,
            created_at=user.created_at.isoformat() if user.created_at else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user account: {str(e)}"
        )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and return JWT access token.

    Uses OAuth2 password flow for form-based authentication.
    Returns a JWT token valid for ACCESS_TOKEN_EXPIRE_MINUTES.

    Args:
        form_data: OAuth2 form with username (email) and password
        db: Database session

    Returns:
        Token with access_token and token_type (bearer)

    Raises:
        HTTPException 401: If credentials are invalid
    """
    auth_service = get_auth_service()

    # Create UserLogin from form data
    login_data = UserLogin(
        email=form_data.username,
        password=form_data.password
    )

    try:
        token = await auth_service.authenticate_user(login_data, db)
        return token
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/login/json", response_model=Token)
async def login_json(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return JWT access token (JSON endpoint).

    Alternative to OAuth2 form endpoint that accepts JSON.
    Useful for API clients that can't use form encoding.

    Args:
        user_data: Login data with email and password
        db: Database session

    Returns:
        Token with access_token and token_type (bearer)

    Raises:
        HTTPException 401: If credentials are invalid
    """
    auth_service = get_auth_service()

    try:
        token = await auth_service.authenticate_user(user_data, db)
        return token
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get information about the currently authenticated user.

    Requires valid JWT token in Authorization header.

    Args:
        current_user: Authenticated user from dependency
        db: Database session

    Returns:
        UserResponse with user data
    """
    auth_service = get_auth_service()
    return await auth_service.get_user_by_id(current_user.id, db)


@router.post("/change-password")
async def change_password(
    change_data: UserChangePassword,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Change the authenticated user's password.

    Requires current password for verification and validates new password strength.

    Args:
        change_data: Password change data with current and new passwords
        current_user: Authenticated user from dependency
        db: Database session

    Returns:
        Dict with success message

    Raises:
        HTTPException 400: If current password is incorrect or new password is weak
    """
    auth_service = get_auth_service()

    try:
        await auth_service.change_password(
            current_user.id,
            change_data.current_password,
            change_data.new_password,
            db
        )
        return {"message": "Password changed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to change password: {str(e)}"
        )


@router.put("/me", response_model=UserResponse)
async def update_user_info(
    update_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update the authenticated user's information.

    Allows updating display name and daily budget limit.

    Args:
        update_data: User update data
        current_user: Authenticated user from dependency
        db: Database session

    Returns:
        UserResponse with updated user data

    Raises:
        HTTPException 400: If validation fails
    """
    auth_service = get_auth_service()

    try:
        updated_user = await auth_service.update_user(
            current_user.id,
            update_data.display_name,
            update_data.daily_budget_limit,
            db
        )
        return UserResponse(
            id=str(updated_user.id),
            email=updated_user.email,
            display_name=updated_user.display_name,
            is_active=updated_user.is_active,
            daily_budget_limit=updated_user.daily_budget_limit,
            created_at=updated_user.created_at.isoformat() if updated_user.created_at else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )
