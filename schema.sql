-- ArXiv Metadata Database Schema
-- This schema stores metadata from arXiv papers and includes a path to PDF files

CREATE TABLE IF NOT EXISTS arxiv_papers (
    -- Primary key: arXiv paper ID
    id VARCHAR(50) PRIMARY KEY,

    -- Submitter information
    submitter VARCHAR(500),

    -- Author information
    authors TEXT,
    authors_parsed JSONB,

    -- Paper details
    title TEXT NOT NULL,
    abstract TEXT,
    comments TEXT,

    -- Publication details
    journal_ref TEXT,
    doi VARCHAR(200),
    report_no TEXT,

    -- Classification (kept for reference, but use arxiv_paper_categories for queries)
    categories TEXT,

    -- License information
    license TEXT,

    -- Version history
    versions JSONB,

    -- Dates
    update_date DATE,

    -- PDF file path on external SSD
    pdf_path TEXT,

    -- Metadata timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Category vocabulary table
CREATE TABLE IF NOT EXISTS arxiv_categories (
    category TEXT PRIMARY KEY,
    label TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Junction table: many-to-many relationship between papers and categories
CREATE TABLE IF NOT EXISTS arxiv_paper_categories (
    paper_id VARCHAR(50) NOT NULL
        REFERENCES arxiv_papers(id) ON DELETE CASCADE,
    category TEXT NOT NULL
        REFERENCES arxiv_categories(category) ON UPDATE CASCADE,
    PRIMARY KEY (paper_id, category)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_arxiv_update_date ON arxiv_papers(update_date);
CREATE INDEX IF NOT EXISTS idx_arxiv_authors ON arxiv_papers USING gin(to_tsvector('english', authors));
CREATE INDEX IF NOT EXISTS idx_arxiv_title ON arxiv_papers USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_arxiv_abstract ON arxiv_papers USING gin(to_tsvector('english', abstract));

-- Indexes for category queries (many-to-many)
CREATE INDEX IF NOT EXISTS idx_apc_category ON arxiv_paper_categories(category);
CREATE INDEX IF NOT EXISTS idx_apc_paper ON arxiv_paper_categories(paper_id);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_arxiv_papers_updated_at BEFORE UPDATE
    ON arxiv_papers FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments to document the schema
COMMENT ON TABLE arxiv_papers IS 'ArXiv paper metadata with PDF file paths';
COMMENT ON COLUMN arxiv_papers.id IS 'ArXiv paper identifier (e.g., 0704.0001)';
COMMENT ON COLUMN arxiv_papers.pdf_path IS 'Full path to PDF file on configured storage location (see PDF_BASE_PATH in .env)';
COMMENT ON COLUMN arxiv_papers.versions IS 'JSON array of version history';
COMMENT ON COLUMN arxiv_papers.authors_parsed IS 'JSON array of parsed author names';
COMMENT ON COLUMN arxiv_papers.categories IS 'Raw space-separated categories from source (for reference only)';

COMMENT ON TABLE arxiv_categories IS 'Category vocabulary for arXiv papers';
COMMENT ON TABLE arxiv_paper_categories IS 'Many-to-many junction table linking papers to categories';
