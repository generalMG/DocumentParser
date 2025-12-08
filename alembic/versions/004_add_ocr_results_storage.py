"""Add OCR results storage column

Revision ID: 004
Revises: 003
Create Date: 2025-12-08

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add column to store OCR results as JSONB"""

    # Add ocr_results column (JSONB to store OCR output)
    op.add_column(
        'arxiv_papers',
        sa.Column('ocr_results', JSONB, nullable=True)
    )

    # Add tracking columns for OCR processing
    op.add_column(
        'arxiv_papers',
        sa.Column('ocr_processed', sa.Boolean, server_default='false', nullable=False)
    )
    op.add_column(
        'arxiv_papers',
        sa.Column('ocr_processed_at', sa.TIMESTAMP, nullable=True)
    )
    op.add_column(
        'arxiv_papers',
        sa.Column('ocr_error', sa.Text, nullable=True)
    )

    # Create index for finding papers needing OCR processing
    op.create_index('idx_arxiv_ocr_processed', 'arxiv_papers', ['ocr_processed'])


def downgrade() -> None:
    """Remove OCR results storage columns"""

    op.drop_index('idx_arxiv_ocr_processed', table_name='arxiv_papers')
    op.drop_column('arxiv_papers', 'ocr_error')
    op.drop_column('arxiv_papers', 'ocr_processed_at')
    op.drop_column('arxiv_papers', 'ocr_processed')
    op.drop_column('arxiv_papers', 'ocr_results')
