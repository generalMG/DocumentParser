"""
SQLAlchemy models for ArXiv database
"""

from sqlalchemy import Column, String, Text, Date, TIMESTAMP, Boolean, ForeignKey, Index, func, LargeBinary
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class ArxivPaper(Base):
    """Main table storing arXiv paper metadata"""
    __tablename__ = 'arxiv_papers'

    # Primary key
    id = Column(String(50), primary_key=True)

    # Submitter information
    submitter = Column(String(500))

    # Author information
    authors = Column(Text)
    authors_parsed = Column(JSONB)

    # Paper details
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    comments = Column(Text)

    # Publication details
    journal_ref = Column(Text)
    doi = Column(String(200))
    report_no = Column(Text)

    # Classification (kept for reference, use paper_categories relationship for queries)
    categories = Column(Text)

    # License information
    license = Column(Text)

    # Version history
    versions = Column(JSONB)

    # Dates
    update_date = Column(Date)

    # PDF file path (see PDF_BASE_PATH in .env)
    pdf_path = Column(Text)

    # PDF binary content (stored in database)
    pdf_content = Column(LargeBinary, nullable=True)

    # PDF download tracking
    pdf_downloaded = Column(Boolean, default=False, server_default='false')
    pdf_download_attempted_at = Column(TIMESTAMP, nullable=True)
    pdf_download_error = Column(Text, nullable=True)

    # Metadata timestamps
    created_at = Column(TIMESTAMP, default=datetime.utcnow, server_default=func.now())
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, server_default=func.now())

    # Relationships
    paper_categories = relationship(
        'ArxivPaperCategory',
        back_populates='paper',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f"<ArxivPaper(id='{self.id}', title='{self.title[:50]}...')>"


class ArxivCategory(Base):
    """Category vocabulary table"""
    __tablename__ = 'arxiv_categories'

    category = Column(Text, primary_key=True)
    label = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, server_default=func.now())

    # Relationships
    paper_categories = relationship(
        'ArxivPaperCategory',
        back_populates='category_obj'
    )

    def __repr__(self):
        return f"<ArxivCategory(category='{self.category}')>"


class ArxivPaperCategory(Base):
    """Junction table for many-to-many relationship between papers and categories"""
    __tablename__ = 'arxiv_paper_categories'

    paper_id = Column(
        String(50),
        ForeignKey('arxiv_papers.id', ondelete='CASCADE'),
        primary_key=True
    )
    category = Column(
        Text,
        ForeignKey('arxiv_categories.category', onupdate='CASCADE'),
        primary_key=True
    )

    # Relationships
    paper = relationship('ArxivPaper', back_populates='paper_categories')
    category_obj = relationship('ArxivCategory', back_populates='paper_categories')

    def __repr__(self):
        return f"<ArxivPaperCategory(paper_id='{self.paper_id}', category='{self.category}')>"


# Create indexes
Index('idx_arxiv_update_date', ArxivPaper.update_date)
Index('idx_apc_category', ArxivPaperCategory.category)
Index('idx_apc_paper', ArxivPaperCategory.paper_id)

# Full-text search indexes will be created in migration
