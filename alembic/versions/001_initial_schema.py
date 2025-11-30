"""Initial schema for ArXiv database

Revision ID: 001
Revises:
Create Date: 2025-01-05

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables and indexes"""

    # Create arxiv_papers table
    op.create_table(
        'arxiv_papers',
        sa.Column('id', sa.String(50), primary_key=True),
        sa.Column('submitter', sa.String(500)),
        sa.Column('authors', sa.Text),
        sa.Column('authors_parsed', JSONB),
        sa.Column('title', sa.Text, nullable=False),
        sa.Column('abstract', sa.Text),
        sa.Column('comments', sa.Text),
        sa.Column('journal_ref', sa.Text),
        sa.Column('doi', sa.String(200)),
        sa.Column('report_no', sa.Text),
        sa.Column('categories', sa.Text),
        sa.Column('license', sa.Text),
        sa.Column('versions', JSONB),
        sa.Column('update_date', sa.Date),
        sa.Column('pdf_path', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now()),
        comment='ArXiv paper metadata with PDF file paths'
    )

    # Create arxiv_categories table
    op.create_table(
        'arxiv_categories',
        sa.Column('category', sa.Text, primary_key=True),
        sa.Column('label', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        comment='Category vocabulary for arXiv papers'
    )

    # Create arxiv_paper_categories junction table
    op.create_table(
        'arxiv_paper_categories',
        sa.Column('paper_id', sa.String(50), nullable=False),
        sa.Column('category', sa.Text, nullable=False),
        sa.PrimaryKeyConstraint('paper_id', 'category'),
        sa.ForeignKeyConstraint(
            ['paper_id'],
            ['arxiv_papers.id'],
            ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['category'],
            ['arxiv_categories.category'],
            onupdate='CASCADE'
        ),
        comment='Many-to-many junction table linking papers to categories'
    )

    # Create standard indexes
    op.create_index('idx_arxiv_update_date', 'arxiv_papers', ['update_date'])
    op.create_index('idx_apc_category', 'arxiv_paper_categories', ['category'])
    op.create_index('idx_apc_paper', 'arxiv_paper_categories', ['paper_id'])

    # Create full-text search indexes using GIN
    op.execute("""
        CREATE INDEX idx_arxiv_authors
        ON arxiv_papers
        USING gin(to_tsvector('english', authors))
    """)

    op.execute("""
        CREATE INDEX idx_arxiv_title
        ON arxiv_papers
        USING gin(to_tsvector('english', title))
    """)

    op.execute("""
        CREATE INDEX idx_arxiv_abstract
        ON arxiv_papers
        USING gin(to_tsvector('english', abstract))
    """)

    # Create trigger function for updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    # Create trigger
    op.execute("""
        CREATE TRIGGER update_arxiv_papers_updated_at
        BEFORE UPDATE ON arxiv_papers
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    """Drop all tables and indexes"""

    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS update_arxiv_papers_updated_at ON arxiv_papers")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")

    # Drop indexes (full-text)
    op.execute("DROP INDEX IF EXISTS idx_arxiv_authors")
    op.execute("DROP INDEX IF EXISTS idx_arxiv_title")
    op.execute("DROP INDEX IF EXISTS idx_arxiv_abstract")

    # Drop standard indexes
    op.drop_index('idx_apc_paper', 'arxiv_paper_categories')
    op.drop_index('idx_apc_category', 'arxiv_paper_categories')
    op.drop_index('idx_arxiv_update_date', 'arxiv_papers')

    # Drop tables (in reverse order due to foreign keys)
    op.drop_table('arxiv_paper_categories')
    op.drop_table('arxiv_categories')
    op.drop_table('arxiv_papers')
