"""Add PDF binary storage column

Revision ID: 003
Revises: 002
Create Date: 2025-12-04

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add column to store PDF binary content"""

    # Add pdf_content column (BYTEA to store PDF files in database)
    op.add_column(
        'arxiv_papers',
        sa.Column('pdf_content', sa.LargeBinary, nullable=True)
    )


def downgrade() -> None:
    """Remove PDF binary storage column"""

    # Drop column
    op.drop_column('arxiv_papers', 'pdf_content')
