"""Tests for authentication module."""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, get_db
from backend.models import User
from backend.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token,
    validate_password_strength,
    sanitize_string,
    sanitize_email,
    Token,
    TokenData,
    UserCreate,
    UserLogin,
    UserResponse,
    UserChangePassword,
    UserUpdate,
)
from backend.auth.dependencies import get_current_user, get_current_active_user, get_optional_user
from backend.auth.service import AuthService, get_auth_service


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


class TestPasswordFunctions:
    """Tests for password hashing and verification."""

    def test_hash_and_verify_password(self):
        """Test password hashing and verification."""
        password = "Secure123!"
        hashed = get_password_hash(password)

        # Hash should be different from original
        assert hashed != password

        # Verify correct password
        assert verify_password(password, hashed) is True

        # Verify incorrect password
        assert verify_password("Wrong123!", hashed) is False

    def test_hash_different_for_same_password(self):
        """Test that hashing same password twice produces different hashes."""
        password = "Test456@"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # Hashes should be different (bcrypt uses random salt)
        assert hash1 != hash2

        # But both should verify correctly
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


class TestJWTToken:
    """Tests for JWT token creation and verification."""

    def test_create_and_decode_token(self):
        """Test creating and decoding a JWT token."""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

        # Decode token
        token_data = decode_access_token(token)
        assert token_data is not None
        assert token_data.email == "test@example.com"
        assert token_data.exp is not None

    def test_decode_invalid_token(self):
        """Test decoding an invalid token."""
        invalid_token = "invalid.token.string"
        token_data = decode_access_token(invalid_token)

        assert token_data is None

    def test_decode_expired_token(self):
        """Test decoding an expired token.

        Our decode function uses jwt.decode() which validates expiration
        by default. Expired tokens should return None.
        """
        # Create token that's already expired
        data = {"sub": "test@example.com"}
        expired_delta = timedelta(seconds=-60)  # Expired 1 minute ago
        token = create_access_token(data, expires_delta=expired_delta)

        # Token should NOT decode because it's expired
        # jwt.decode() validates expiration by default
        token_data = decode_access_token(token)
        assert token_data is None

    def test_token_with_custom_expiration(self):
        """Test creating token with custom expiration."""
        data = {"sub": "test@example.com"}
        custom_expire = timedelta(hours=2)
        token = create_access_token(data, expires_delta=custom_expire)

        token_data = decode_access_token(token)
        assert token_data is not None

        # Check expiration is approximately 2 hours from now
        # Note: this may vary slightly based on test execution time
        time_until_expiry = token_data.exp - datetime.utcnow()
        assert timedelta(seconds=7000) < time_until_expiry < timedelta(seconds=7400)


class TestPasswordValidation:
    """Tests for password strength validation."""

    def test_valid_password(self):
        """Test a valid password passes validation."""
        password = "Secure123!"
        is_valid, errors = validate_password_strength(password)

        assert is_valid is True
        assert len(errors) == 0

    def test_password_too_short(self):
        """Test password that's too short."""
        password = "Short1!"
        is_valid, errors = validate_password_strength(password)

        assert is_valid is False
        assert any("8 characters" in e for e in errors)

    def test_password_too_long(self):
        """Test password that's too long."""
        password = "a" * 101  # 101 characters
        is_valid, errors = validate_password_strength(password)

        assert is_valid is False
        assert any("100 characters" in e for e in errors)

    def test_password_missing_lowercase(self):
        """Test password without lowercase letters."""
        password = "UPPERCASE123!"
        is_valid, errors = validate_password_strength(password)

        assert is_valid is False
        assert any("lowercase" in e for e in errors)

    def test_password_missing_uppercase(self):
        """Test password without uppercase letters."""
        password = "lowercase123!"
        is_valid, errors = validate_password_strength(password)

        assert is_valid is False
        assert any("uppercase" in e for e in errors)

    def test_password_missing_digit(self):
        """Test password without digits."""
        password = "NoDigitsHere!"
        is_valid, errors = validate_password_strength(password)

        assert is_valid is False
        assert any("digit" in e for e in errors)

    def test_password_missing_special_char(self):
        """Test password without special characters."""
        password = "NoSpecialChars123"
        is_valid, errors = validate_password_strength(password)

        assert is_valid is False
        assert any("special character" in e for e in errors)

    def test_all_errors_at_once(self):
        """Test password that fails all validations."""
        password = "short"
        is_valid, errors = validate_password_strength(password)

        assert is_valid is False
        assert len(errors) > 1  # Should have multiple errors


class TestSanitization:
    """Tests for input sanitization."""

    def test_sanitize_string_basic(self):
        """Test basic string sanitization."""
        input_str = "Hello World"
        result = sanitize_string(input_str)

        assert result == "Hello World"

    def test_sanitize_string_truncation(self):
        """Test string truncation."""
        input_str = "a" * 2000
        result = sanitize_string(input_str, max_length=100)

        assert len(result) == 100

    def test_sanitize_string_null_bytes(self):
        """Test null byte removal."""
        input_str = "Hello\x00World"
        result = sanitize_string(input_str)

        assert "\x00" not in result
        assert result == "HelloWorld"

    def test_sanitize_string_sql_patterns(self):
        """Test SQL pattern removal."""
        input_str = "Hello'; DROP TABLE users; --"
        result = sanitize_string(input_str)

        assert "--" not in result
        assert ";--" not in result

    def test_sanitize_empty_string(self):
        """Test empty string handling."""
        result = sanitize_string("")
        assert result == ""

    def test_sanitize_none(self):
        """Test None handling."""
        result = sanitize_string(None)
        assert result == ""

    def test_sanitize_email_basic(self):
        """Test basic email sanitization."""
        email = "Test@Example.COM"
        result = sanitize_email(email)

        assert result == "test@example.com"

    def test_sanitize_email_trimming(self):
        """Test email trimming."""
        email = "  test@example.com  "
        result = sanitize_email(email)

        assert result == "test@example.com"

    def test_sanitize_email_special_chars(self):
        """Test email special character handling."""
        email = "test+user@example.com"
        result = sanitize_email(email)

        # Plus sign should be removed by our sanitization
        assert "+" not in result

    def test_sanitize_email_multiple_at(self):
        """Test email with multiple @ symbols."""
        email = "test@user@example.com"
        result = sanitize_email(email)

        # Should keep first and last parts
        assert result == "test@example.com"


class TestPydanticModels:
    """Tests for Pydantic models."""

    def test_token_model(self):
        """Test Token model."""
        token = Token(access_token="test_token", token_type="bearer")

        assert token.access_token == "test_token"
        assert token.token_type == "bearer"

    def test_token_data_model(self):
        """Test TokenData model."""
        now = datetime.utcnow()
        token_data = TokenData(email="test@example.com", exp=now)

        assert token_data.email == "test@example.com"
        assert token_data.exp == now

    def test_user_create_valid(self):
        """Test UserCreate with valid data."""
        user = UserCreate(
            email="test@example.com",
            password="Secure123!",
            display_name="Test User"
        )

        assert user.email == "test@example.com"
        assert user.password == "Secure123!"
        assert user.display_name == "Test User"

    def test_user_create_password_too_short(self):
        """Test UserCreate rejects short password."""
        with pytest.raises(ValueError):
            UserCreate(
                email="test@example.com",
                password="short",
                display_name="Test User"
            )

    def test_user_login_model(self):
        """Test UserLogin model."""
        login = UserLogin(
            email="test@example.com",
            password="Secure123!"
        )

        assert login.email == "test@example.com"
        assert login.password == "Secure123!"

    def test_user_response_model(self):
        """Test UserResponse model."""
        user_response = UserResponse(
            id="123",
            email="test@example.com",
            display_name="Test User",
            daily_budget_limit=10.0,
            is_active=True,
            created_at="2024-01-01T00:00:00Z"
        )

        assert user_response.id == "123"
        assert user_response.email == "test@example.com"
        assert user_response.display_name == "Test User"
        assert user_response.daily_budget_limit == 10.0

    def test_user_change_password_model(self):
        """Test UserChangePassword model."""
        change = UserChangePassword(
            current_password="Old12345!",
            new_password="New67890!"
        )

        assert change.current_password == "Old12345!"
        assert change.new_password == "New67890!"

    def test_user_update_model(self):
        """Test UserUpdate model."""
        update = UserUpdate(
            display_name="Updated Name",
            daily_budget_limit=20.0
        )

        assert update.display_name == "Updated Name"
        assert update.daily_budget_limit == 20.0

    def test_user_update_invalid_budget(self):
        """Test UserUpdate rejects invalid budget."""
        # Negative budget
        with pytest.raises(ValueError):
            UserUpdate(daily_budget_limit=-10.0)

        # Too large budget
        with pytest.raises(ValueError):
            UserUpdate(daily_budget_limit=20000.0)


class TestAuthService:
    """Tests for AuthService."""

    @pytest.mark.asyncio
    async def test_register_user(self, db_session):
        """Test user registration."""
        user_data = UserCreate(
            email="test@example.com",
            password="Secure123!",
            display_name="Test User"
        )

        auth_service = get_auth_service()
        user_response = await auth_service.register_user(user_data, db_session)

        # Verify the response
        assert user_response.email == "test@example.com"
        assert user_response.display_name == "Test User"
        assert user_response.is_active is True

        # Query the database to verify the password was hashed correctly
        from sqlalchemy import select
        result = await db_session.execute(select(User).where(User.email == "test@example.com"))
        user = result.scalar_one()
        assert verify_password("Secure123!", user.hashed_password)

    @pytest.mark.asyncio
    async def test_register_user_duplicate_email(self, db_session):
        """Test registration with duplicate email fails."""
        user_data = UserCreate(
            email="test@example.com",
            password="Secure123!"
        )

        auth_service = get_auth_service()
        await auth_service.register_user(user_data, db_session)

        # Try to register again with same email
        with pytest.raises(Exception):  # HTTPException 409
            await auth_service.register_user(user_data, db_session)

    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, db_session):
        """Test successful user authentication."""
        # First register a user
        user_data = UserCreate(
            email="test@example.com",
            password="Secure123!"
        )

        auth_service = get_auth_service()
        await auth_service.register_user(user_data, db_session)

        # Now authenticate
        login_data = UserLogin(
            email="test@example.com",
            password="Secure123!"
        )
        token = await auth_service.authenticate_user(login_data, db_session)

        assert token is not None
        assert token.access_token is not None
        assert token.token_type == "bearer"

    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(self, db_session):
        """Test authentication with wrong password fails."""
        # First register a user
        user_data = UserCreate(
            email="test@example.com",
            password="Secure123!"
        )

        auth_service = get_auth_service()
        await auth_service.register_user(user_data, db_session)

        # Try to authenticate with wrong password
        login_data = UserLogin(
            email="test@example.com",
            password="Wrong1234!"
        )

        with pytest.raises(Exception):  # HTTPException 401
            await auth_service.authenticate_user(login_data, db_session)

    @pytest.mark.asyncio
    async def test_authenticate_user_nonexistent(self, db_session):
        """Test authentication with non-existent user fails."""
        auth_service = get_auth_service()

        login_data = UserLogin(
            email="nonexistent@example.com",
            password="Some1234!"
        )

        with pytest.raises(Exception):  # HTTPException 401
            await auth_service.authenticate_user(login_data, db_session)

    @pytest.mark.asyncio
    async def test_get_user_by_id(self, db_session):
        """Test fetching user by ID."""
        # Register a user
        user_data = UserCreate(
            email="test@example.com",
            password="Secure123!"
        )

        auth_service = get_auth_service()
        created_user = await auth_service.register_user(user_data, db_session)

        # Fetch user by ID
        fetched_user = await auth_service.get_user_by_id(created_user.id, db_session)

        assert fetched_user is not None
        assert fetched_user.id == str(created_user.id)
        assert fetched_user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_update_user(self, db_session):
        """Test updating user information."""
        # Register a user
        user_data = UserCreate(
            email="test@example.com",
            password="Secure123!"
        )

        auth_service = get_auth_service()
        user = await auth_service.register_user(user_data, db_session)

        # Update user
        updated_user = await auth_service.update_user(
            user.id,
            display_name="Updated Name",
            daily_budget_limit=25.0,
            db=db_session
        )

        assert updated_user.display_name == "Updated Name"
        assert updated_user.daily_budget_limit == 25.0

    @pytest.mark.asyncio
    async def test_change_password(self, db_session):
        """Test changing password."""
        # Register a user
        user_data = UserCreate(
            email="test@example.com",
            password="Old12345!"
        )

        auth_service = get_auth_service()
        user_response = await auth_service.register_user(user_data, db_session)

        # Store the hashed password to verify it changed
        from sqlalchemy import select
        result = await db_session.execute(select(User).where(User.id == user_response.id))
        user = result.scalar_one()
        old_hash = user.hashed_password

        # Change password
        await auth_service.change_password(
            user.id,
            "Old12345!",
            "New67890!",
            db_session
        )

        # Refresh user from DB
        result = await db_session.execute(select(User).where(User.id == user.id))
        user = result.scalar_one()

        # Verify password hash changed
        assert user.hashed_password != old_hash

        # Verify new password works
        assert verify_password("New67890!", user.hashed_password)

    @pytest.mark.asyncio
    async def test_change_password_wrong_current(self, db_session):
        """Test changing password with wrong current password fails."""
        # Register a user
        user_data = UserCreate(
            email="test@example.com",
            password="Correct123!"
        )

        auth_service = get_auth_service()
        user = await auth_service.register_user(user_data, db_session)

        # Try to change with wrong current password
        with pytest.raises(Exception):  # HTTPException 401
            await auth_service.change_password(
                user.id,
                "Wrong123!",
                "New456!",
                db_session
            )


class TestDependencies:
    """Tests for FastAPI dependencies."""

    @pytest.mark.asyncio
    async def test_get_optional_user_no_token(self, db_session):
        """Test get_optional_user returns None when no token."""
        from fastapi.security import HTTPAuthorizationCredentials
        import unittest.mock as mock

        # Mock credentials with no token (None)
        mock_creds = mock.Mock(spec=HTTPAuthorizationCredentials)
        mock_creds.credentials = None

        result = await get_optional_user(
            credentials=mock_creds,
            db=db_session
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_optional_user_invalid_token(self, db_session):
        """Test get_optional_user returns None for invalid token."""
        from fastapi.security import HTTPAuthorizationCredentials
        import unittest.mock as mock

        # Mock credentials with invalid token
        mock_creds = mock.Mock(spec=HTTPAuthorizationCredentials)
        mock_creds.credentials = "invalid.token.string"

        result = await get_optional_user(
            credentials=mock_creds,
            db=db_session
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_optional_user_valid_token_nonexistent_user(self, db_session):
        """Test get_optional_user returns None for valid token but nonexistent user."""
        from fastapi.security import HTTPAuthorizationCredentials
        import unittest.mock as mock

        # Create a valid token
        token = create_access_token({"sub": "nonexistent@example.com"})

        mock_creds = mock.Mock(spec=HTTPAuthorizationCredentials)
        mock_creds.credentials = token

        result = await get_optional_user(
            credentials=mock_creds,
            db=db_session
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self, db_session):
        """Test get_current_user with valid token."""
        # Register a user first
        user_data = UserCreate(
            email="test@example.com",
            password="Secure123!"
        )

        auth_service = get_auth_service()
        await auth_service.register_user(user_data, db_session)

        # Create token
        token = create_access_token({"sub": "test@example.com"})

        # Mock the credentials
        from fastapi.security import HTTPAuthorizationCredentials
        from fastapi import HTTPException
        import unittest.mock as mock

        mock_creds = mock.Mock(spec=HTTPAuthorizationCredentials)
        mock_creds.credentials = token

        # Get the user
        result_user = await get_current_user(
            credentials=mock_creds,
            db=db_session
        )

        assert result_user is not None
        assert result_user.email == "test@example.com"
