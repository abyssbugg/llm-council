"""Authentication module for JWT and password handling."""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

from ..config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Data extracted from JWT token."""
    email: str
    exp: Optional[datetime] = None


class UserCreate(BaseModel):
    """User registration schema."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    display_name: Optional[str] = Field(None, max_length=255)


class UserLogin(BaseModel):
    """User login schema."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response schema."""
    id: str
    email: str
    display_name: Optional[str] = None
    daily_budget_limit: float = 10.0
    is_active: bool = True
    created_at: Optional[str] = None


class UserChangePassword(BaseModel):
    """Change password schema."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseModel):
    """User update schema."""
    display_name: Optional[str] = Field(None, max_length=255)
    daily_budget_limit: Optional[float] = Field(None, ge=0, le=10000)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.
    
    Args:
        plain_password: Plain text password to verify.
        hashed_password: Hashed password to compare against.
    
    Returns:
        True if password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password.
    
    Args:
        password: Plain text password to hash.
    
    Returns:
        Hashed password.
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.
    
    Args:
        data: Data to encode in the token.
        expires_delta: Optional expiration time override.
    
    Returns:
        Encoded JWT token.
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and verify a JWT access token.
    
    Args:
        token: JWT token to decode.
    
    Returns:
        TokenData if valid, None otherwise.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        
        exp = payload.get("exp")
        if exp is not None:
            from datetime import timezone
            exp = datetime.fromtimestamp(exp, tz=timezone.utc)
        
        return TokenData(email=email, exp=exp)
    
    except JWTError:
        return None


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """Validate password strength.
    
    Args:
        password: Password to validate.
    
    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if len(password) > 100:
        errors.append("Password must be no more than 100 characters long")
    
    # Check for at least one lowercase letter
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    # Check for at least one uppercase letter
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    # Check for at least one digit
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")
    
    # Check for at least one special character
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if not any(c in special_chars for c in password):
        errors.append("Password must contain at least one special character")
    
    return len(errors) == 0, errors


def sanitize_string(input_string: str, max_length: int = 1000) -> str:
    """Sanitize a string input to prevent injection attacks.
    
    Args:
        input_string: String to sanitize.
        max_length: Maximum allowed length.
    
    Returns:
        Sanitized string.
    """
    if not input_string:
        return ""
    
    # Truncate to max length
    result = input_string[:max_length]
    
    # Remove null bytes
    result = result.replace("\x00", "")
    
    # Remove potentially dangerous SQL patterns (basic)
    dangerous_patterns = ["--", ";--", "/*", "*/"]
    for pattern in dangerous_patterns:
        result = result.replace(pattern, "")
    
    return result.strip()


def sanitize_email(email: str) -> str:
    """Sanitize an email address.
    
    Args:
        email: Email address to sanitize.
    
    Returns:
        Sanitized email address (lowercase).
    """
    if not email:
        return ""
    
    # Basic email sanitization
    email = email.strip().lower()
    
    # Remove any characters that shouldn't be in an email
    allowed_chars = "abcdefghijklmnopqrstuvwxyz0123456789@.-_"
    email = "".join(c for c in email if c in allowed_chars)
    
    # Ensure only one @ symbol
    parts = email.split("@")
    if len(parts) == 2:
        return "@".join(parts)
    elif len(parts) > 2:
        # Keep first and last parts
        return f"{parts[0]}@{parts[-1]}"
    
    return email
