"""Add PDF download tracking columns

Revision ID: 002
Revises: 001
Create Date: 2025-01-08

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add columns to track PDF download status"""

    # Add pdf_downloaded column (boolean to track if PDF was successfully downloaded)
    op.add_column(
        'arxiv_papers',
        sa.Column('pdf_downloaded', sa.Boolean, server_default='false', nullable=False)
    )

    # Add pdf_download_attempted_at column (timestamp of last download attempt)
    op.add_column(
        'arxiv_papers',
        sa.Column('pdf_download_attempted_at', sa.TIMESTAMP, nullable=True)
    )

    # Add pdf_download_error column (stores error message if download failed)
    op.add_column(
        'arxiv_papers',
        sa.Column('pdf_download_error', sa.Text, nullable=True)
    )

    # Create index on pdf_downloaded for faster queries
    op.create_index(
        'idx_arxiv_pdf_downloaded',
        'arxiv_papers',
        ['pdf_downloaded']
    )


def downgrade() -> None:
    """Remove PDF download tracking columns"""

    # Drop index
    op.drop_index('idx_arxiv_pdf_downloaded', 'arxiv_papers')

    # Drop columns
    op.drop_column('arxiv_papers', 'pdf_download_error')
    op.drop_column('arxiv_papers', 'pdf_download_attempted_at')
    op.drop_column('arxiv_papers', 'pdf_downloaded')
