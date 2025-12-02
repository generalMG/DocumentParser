# ArXiv Database Project

> **Note**: This project is for **testing and learning purposes only**. It demonstrates data pipeline architecture, database design, and Python data processing techniques.

A Python-based system for analyzing and managing arXiv metadata with PostgreSQL storage and PDF download capabilities.

## Overview

This project provides tools to:
- Analyze arXiv metadata from JSON snapshots
- Store metadata in PostgreSQL with efficient indexing
- Manage PDF downloads from arXiv
- Track download status and handle rate limiting

## Features

- **Fast parallel analysis** of arXiv metadata (1.7M+ records)
- **PostgreSQL database** with optimized schema and full-text search
- **SQLAlchemy ORM** with declarative models
- **Alembic migrations** for version-controlled schema management
- **Environment-based configuration** for portability
- **Connection pooling** and transaction management
- **Unix socket support** for local development
- **PDF download tracking** with rate limiting

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- ~5GB disk space for metadata
- Additional storage for PDFs (optional)

## Installation

### 1. Clone the repository

\`\`\`bash
git clone <repository-url>
cd arxiv_database
\`\`\`

### 2. Install dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Required packages:
- \`sqlalchemy>=2.0.0\` - ORM and database toolkit
- \`psycopg2-binary\` - PostgreSQL adapter
- \`alembic\` - Database migration tool
- \`python-dotenv\` - Environment variable management
- \`matplotlib\` - Data visualization

### 3. Configure environment

\`\`\`bash
cp .env.example .env
# Edit .env with your database credentials
nano .env
\`\`\`

**Important**: Set \`DB_HOST\` based on your connection type:
- Leave empty (\`DB_HOST=\`) for Unix socket connection (peer authentication)
- Set to \`localhost\` for TCP/IP connection (requires password)

### 4. Set up PostgreSQL database

\`\`\`bash
# Create database
createdb arxiv

# Run migrations (recommended approach)
alembic upgrade head

# Alternative: Apply schema directly (not recommended if using migrations)
# psql -d arxiv -f schema.sql
\`\`\`

## Configuration

All configuration is managed through environment variables in \`.env\`:

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| \`DB_HOST\` | PostgreSQL host | _(empty)_ | Empty = Unix socket, \`localhost\` = TCP/IP |
| \`DB_PORT\` | PostgreSQL port | \`5432\` | Standard PostgreSQL port |
| \`DB_NAME\` | Database name | \`arxiv\` | Database to connect to |
| \`DB_USER\` | Database user | \`postgres\` | Your PostgreSQL username |
| \`DB_PASSWORD\` | Database password | _(empty)_ | Leave empty for peer auth |
| \`SQLALCHEMY_URL\` | Full database URL | \`postgresql://user@/dbname\` | Alternative to individual vars |
| \`PDF_BASE_PATH\` | PDF storage directory | \`/path/to/arxiv_pdfs\` | Adjust to your storage location |
| \`ARXIV_DATA_PATH\` | Path to metadata JSON | \`./arxiv-metadata-oai-snapshot.json\` | Download from Kaggle |
| \`DOWNLOAD_DELAY\` | Delay between downloads (seconds) | \`3.0\` | Respect arXiv rate limits |
| \`CPU_COUNT\` | CPU cores for processing | \`4\` | Adjust based on your system |

### Database Connection Types

**Unix Socket (Recommended for local development)**:
\`\`\`ini
DB_HOST=
SQLALCHEMY_URL=postgresql://your_user@/arxiv
\`\`\`
- Uses peer authentication (no password needed if user matches system user)
- Faster than TCP/IP
- More secure for local development

**TCP/IP**:
\`\`\`ini
DB_HOST=localhost
SQLALCHEMY_URL=postgresql://your_user:password@localhost:5432/arxiv
\`\`\`
- Requires password authentication
- Needed for remote connections

## Usage

### Analyze ArXiv Metadata

\`\`\`bash
# Basic analysis with default settings
python fast_analysis.py

# Specify custom file path
python fast_analysis.py /path/to/arxiv-metadata-oai-snapshot.json

# Use specific number of CPU cores
python fast_analysis.py --cpu_count 8
\`\`\`

**Output**: Generates visualizations in \`outputs/\` showing articles by year and version distribution.

### Database Migrations

\`\`\`bash
# Check current migration version
alembic current

# View migration history
alembic history

# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Create new migration (auto-detect model changes)
alembic revision --autogenerate -m "Description of changes"
\`\`\`

### Load Data into Database

\`\`\`bash
# Test with 1,000 records first
python scripts/load_arxiv_data.py --max-records 1000

# Load all records (1.7M+ papers) - uses settings from .env
python scripts/load_arxiv_data.py

# Custom file and batch size
python scripts/load_arxiv_data.py --json-file /path/to/arxiv-metadata.json --batch-size 5000

# Override database settings
python scripts/load_arxiv_data.py --db-host localhost --db-user custom_user
\`\`\`

**Features**:
- Batch processing (default: 1000 records/batch)
- UPSERT operations (updates existing records, preserves PDF tracking status)
- Progress tracking with records/sec rate
- Automatic category extraction and linking
- Error handling with retry logic

**Output**: Real-time progress display showing processed records, insertion rate, and errors.

### Database Operations

\`\`\`python
from database import DatabaseManager, ArxivPaper, ArxivCategory

# Create database manager (defaults to Unix socket when DB_HOST is empty)
db_manager = DatabaseManager()

# Use session context manager
with db_manager.session_scope() as session:
    # Query papers
    papers = session.query(ArxivPaper).filter(
        ArxivPaper.title.contains('machine learning')
    ).limit(10).all()

    # Add new category
    category = ArxivCategory(category='cs.AI', label='Artificial Intelligence')
    session.add(category)
    # Auto-commits on success, rolls back on exception

# Query papers by PDF download status
with db_manager.session_scope() as session:
    # Get papers that haven't been downloaded yet
    pending = session.query(ArxivPaper).filter(
        ArxivPaper.pdf_downloaded == False
    ).limit(100).all()

    # Get papers with download errors
    failed = session.query(ArxivPaper).filter(
        ArxivPaper.pdf_download_error.isnot(None)
    ).all()
\`\`\`

### OCR (Experimental)

- Install optional deps: `pip install paddlepaddle-gpu paddleocr` (or `paddlepaddle` for CPU-only).
- Run test script on a PDF: `python scripts/test_paddleocr_vl.py /path/to/file.pdf --pages 1 --cpu` (omit `--cpu` to use GPU).
- Models download from PaddleOCR hosts; ensure network access or pre-download to the default cache.

## Project Structure

\`\`\`
arxiv_database/
├── alembic/                  # Database migrations
│   ├── versions/
│   │   ├── 001_initial_schema.py
│   │   └── 002_add_pdf_download_tracking.py
│   ├── env.py               # Alembic environment config
│   └── script.py.mako       # Migration template
├── database/                 # Database package
│   ├── __init__.py          # Package exports
│   ├── database.py          # Connection manager & session handling
│   └── models.py            # SQLAlchemy ORM models
├── scripts/                  # Data loading and management scripts
│   ├── __init__.py          # Package initialization
│   └── load_arxiv_data.py   # Load arXiv JSON data into database
├── outputs/                  # Generated analysis outputs
├── .env.example             # Environment configuration template
├── .gitignore               # Git ignore rules
├── alembic.ini              # Alembic configuration
├── fast_analysis.py         # Fast parallel metadata analyzer
├── json_file_analysis.py    # Legacy single-threaded analyzer
├── requirements.txt         # Python dependencies
├── schema.sql               # PostgreSQL schema design (reference)
└── README.md                # This file
\`\`\`

## Database Schema

### Applied Migrations

The database schema is managed through Alembic migrations:

**Migration 001** - Initial schema (applied)
- Created \`arxiv_papers\`, \`arxiv_categories\`, \`arxiv_paper_categories\` tables
- Added full-text search (GIN) indexes on authors, title, abstract
- Created auto-update trigger for \`updated_at\` timestamp

**Migration 002** - PDF download tracking (applied)
- Added \`pdf_downloaded\` boolean field (default: false)
- Added \`pdf_download_attempted_at\` timestamp field
- Added \`pdf_download_error\` text field for error messages
- Created index on \`pdf_downloaded\` for efficient queries

**Current version**: 002 (head)

### Tables

**\`arxiv_papers\`**: Main table storing paper metadata (20 columns)
- **Primary key**: \`id\` (arXiv paper ID, e.g., "0704.0001")
- **Indexes**: Full-text search (GIN) on authors, title, abstract; B-tree on update_date, pdf_downloaded
- **Core fields**: submitter, authors, title, abstract, DOI, categories, versions (JSONB)
- **PDF tracking**:
  - \`pdf_path\` - Full path to PDF file (based on PDF_BASE_PATH)
  - \`pdf_downloaded\` - Boolean flag indicating successful download (indexed)
  - \`pdf_download_attempted_at\` - Timestamp of last download attempt
  - \`pdf_download_error\` - Error message if download failed
- **Timestamps**: \`created_at\`, \`updated_at\` (auto-updated via trigger)

**\`arxiv_categories\`**: Category vocabulary
- **Primary key**: \`category\` (e.g., "cs.AI", "math.CO")
- **Fields**: \`label\`, \`created_at\`

**\`arxiv_paper_categories\`**: Many-to-many junction table
- **Composite primary key**: (\`paper_id\`, \`category\`)
- **Foreign keys**: CASCADE on delete, UPDATE on category change
- **Indexes**: Optimized for category-based queries

### Key Features
- Full-text search with GIN indexes for efficient text queries
- Automatic \`updated_at\` timestamp via PostgreSQL triggers
- JSONB columns for flexible nested data (versions, parsed authors)
- Foreign key constraints with cascading deletes
- Connection pooling (10 connections, 20 overflow)
- PDF download tracking with status, timestamps, and error logging
- Indexed queries for downloaded/pending PDFs

## Database Package

The \`database\` package provides clean abstractions for database operations:

\`\`\`python
from database import DatabaseManager, get_database_url_from_env
from database import Base, ArxivPaper, ArxivCategory, ArxivPaperCategory

# Create manager from environment variables
db_manager = DatabaseManager()

# Or create with explicit parameters
db_manager = DatabaseManager(
    db_host='localhost',
    db_port=5432,
    db_name='arxiv',
    db_user='postgres',
    db_password='your_password'
)

# Use session context manager (auto-commit/rollback)
with db_manager.session_scope() as session:
    paper = ArxivPaper(id='1234.5678', title='Example Paper')
    session.add(paper)
    # Automatically commits on success, rolls back on exception
\`\`\`

**Features**:
- Connection pooling (configurable pool size)
- Context managers for safe transaction handling
- Environment variable support with sensible defaults
- Unix socket and TCP/IP connection support
- Lazy session creation

## Data Source

Download the arXiv metadata snapshot from:
- **Kaggle**: https://www.kaggle.com/datasets/Cornell-University/arxiv
- **File**: \`arxiv-metadata-oai-snapshot.json\` (~5GB)
- **Format**: JSONL (one paper per line)
- **Records**: 1.7M+ papers

## Development Workflow

### Setting Up for Development

1. Fork and clone the repository
2. Create a virtual environment: \`python -m venv venv\`
3. Activate: \`source venv/bin/activate\` (Linux/Mac) or \`venv\\Scripts\\activate\` (Windows)
4. Install dependencies: \`pip install -r requirements.txt\`
5. Configure \`.env\` file
6. Run migrations: \`alembic upgrade head\`

### Making Schema Changes

1. Modify models in \`database/models.py\`
2. Generate migration: \`alembic revision --autogenerate -m "Description"\`
3. Review generated migration in \`alembic/versions/\`
4. Apply migration: \`alembic upgrade head\`
5. Test rollback: \`alembic downgrade -1\` and \`alembic upgrade head\`

## Development Roadmap

- [x] Environment configuration template
- [x] Database schema design
- [x] Database connection manager
- [x] Git repository with proper .gitignore
- [x] SQLAlchemy ORM models
- [x] Alembic migrations (initial schema + PDF tracking)
- [x] PDF download tracking schema
- [x] Data loading scripts (load arXiv JSON into database)
- [ ] PDF download manager scripts
- [ ] Testing suite
- [ ] API endpoints (optional)
- [ ] Documentation

## Performance

- **Analysis speed**: Processes 1.7M records using multiprocessing
- **Database**: Optimized with GIN indexes for full-text search
- **Connection pooling**: 10 connections, 20 overflow
- **Query performance**: Sub-second lookups with proper indexing

## Troubleshooting

### Connection Issues

**"no password supplied" error**:
- Solution: Set \`DB_HOST=\` (empty) in \`.env\` to use Unix socket

**"peer authentication failed"**:
- Solution: Ensure your system username matches \`DB_USER\`, or use TCP/IP with password

**Alembic can't find models**:
- Solution: Ensure \`.env\` file exists and \`python-dotenv\` is installed

### Migration Issues

**"Target database is not up to date"**:
\`\`\`bash
alembic upgrade head
\`\`\`

**Need to reset migrations**:
\`\`\`bash
alembic downgrade base
alembic upgrade head
\`\`\`

### Build DB With PDFs (Action List)
- Set environment: copy `.env.example` to `.env`, then either set `SQLALCHEMY_URL` directly or configure `DB_HOST/DB_USER/...`. Empty `DB_HOST` uses a Unix socket; `DB_HOST=localhost` uses TCP (needs password).
- Create storage: decide where PDFs live (e.g., `/mnt/d/arxiv_pdfs`) and set `PDF_BASE_PATH` to that path. Ensure the directory exists and is writable.
- Place data: download `arxiv-metadata-oai-snapshot.json` and set `ARXIV_DATA_PATH` (or pass `--json-file`).
- Migrate schema: run `alembic upgrade head` (picks up URL from env or `--db-url`).
- Load metadata: `python scripts/load_arxiv_data.py --db-url "$SQLALCHEMY_URL" --pdf-path "$PDF_BASE_PATH" --json-file "$ARXIV_DATA_PATH" --batch-size 1000 --max-records 5000` (drop `--max-records` to load all). This fills `pdf_path` but leaves `pdf_downloaded` false.
- Mark existing PDFs (optional): if PDFs already exist, update flags (e.g., script to check files and set `pdf_downloaded=true` for matching `pdf_path`).
- Download PDFs: `python scripts/download_pdfs.py --limit 2000 --auto` (respects `DOWNLOAD_DELAY`, updates `pdf_downloaded`, `pdf_download_attempted_at`, `pdf_download_error`). By default it skips rows with a prior `pdf_download_error`; add `--retry-errors` to reattempt failed ones.
- Sync flags to disk state: run `python scripts/sync_pdf_status.py --db-url "$SQLALCHEMY_URL" --pdf-path "$PDF_BASE_PATH"` to align `pdf_downloaded` with what exists on disk.

### Common Pitfalls
- Socket vs TCP: empty `DB_HOST` builds a Unix-socket URL. If Postgres is not on `/var/run/postgresql` or peer auth fails, set `DB_HOST=localhost` (or host) and provide a password.
- Missing data file: loader errors if `arxiv-metadata-oai-snapshot.json` is absent; set `ARXIV_DATA_PATH` or use `--json-file`.
- Disk space: JSON is ~5GB; DB plus PDFs can be tens of GB. Ensure space for both DB storage and `PDF_BASE_PATH`.
- Long loads: all 1.7M rows take time; start with `--max-records` to validate setup.
- Flags not auto-set: loader does not mark existing PDFs as downloaded; run a post-step if needed.

## Notes

- Always respect arXiv's rate limits when downloading PDFs
- Large JSON file not included in repository (add to .gitignore)
- PDF files stored separately, paths tracked in database
- Use Alembic migrations instead of applying schema.sql directly
- Use `scripts/sync_pdf_status.py` to align `pdf_downloaded` flags with files on disk

## License

This project is for educational purposes. See [LICENSE](LICENSE) for details.

## Contributing

This is a learning project. Feel free to fork and experiment!

When contributing:
1. Create a feature branch
2. Make your changes
3. Test migrations: \`alembic upgrade head\` and \`alembic downgrade -1\`
4. Submit a pull request

---

**Disclaimer**: This project is not affiliated with arXiv. Please respect arXiv's [terms of service](https://arxiv.org/help/api/tou) when using their data.
