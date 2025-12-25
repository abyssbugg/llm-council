"""Alembic script template for LLM Council."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import func
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = None
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade database schema."""


def downgrade() -> None:
    """Downgrade database schema."""
