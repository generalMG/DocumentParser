# ArXiv Database Project

> **Note**: This project is for **testing and learning purposes only**. It demonstrates data pipeline architecture, database design, and Python data processing techniques.

A Python-based system for analyzing and managing arXiv metadata with PostgreSQL storage and PDF download capabilities.

## Overview

This project provides tools to:
- Analyze arXiv metadata from JSON snapshots
- Store metadata in PostgreSQL with efficient indexing
- Manage PDF downloads from arXiv
- Track download status and handle rate limiting

## Project Status

Currently migrating from proof-of-concept to production-ready structure with proper database management, version control, and documentation.

## Features

- **Fast parallel analysis** of arXiv metadata (1.7M+ records)
- **PostgreSQL database** with optimized schema and full-text search
- **SQLAlchemy ORM** for database operations
- **Environment-based configuration** for portability
- **Connection pooling** and transaction management
- **PDF download tracking** with rate limiting

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- ~5GB disk space for metadata
- Additional storage for PDFs (optional)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd arxiv_database
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your database credentials
nano .env
```

### 4. Set up PostgreSQL database

```bash
# Create database
createdb arxiv

# Apply schema (optional - will be managed by Alembic migrations later)
psql -d arxiv -f schema.sql
```

## Configuration

All configuration is managed through environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | PostgreSQL host | `localhost` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | Database name | `arxiv` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | _(empty)_ |
| `PDF_BASE_PATH` | PDF storage directory | `/mnt/d/arxiv_pdfs` |
| `ARXIV_DATA_PATH` | Path to metadata JSON | `./arxiv-metadata-oai-snapshot.json` |
| `DOWNLOAD_DELAY` | Delay between downloads (seconds) | `3.0` |
| `CPU_COUNT` | CPU cores for processing | `4` |

## Usage

### Analyze ArXiv Metadata

```bash
# Basic analysis with default settings
python fast_analysis.py

# Specify custom file path
python fast_analysis.py /path/to/arxiv-metadata-oai-snapshot.json

# Use specific number of CPU cores
python fast_analysis.py --cpu_count 8
```

**Output**: Generates visualizations showing articles by year and version distribution.

## Project Structure

```
arxiv_database/
├── database/
│   ├── __init__.py          # Package initialization
│   └── database.py          # Connection manager & session handling
├── outputs/                  # Generated analysis outputs
├── .env.example             # Environment configuration template
├── .gitignore               # Git ignore rules
├── fast_analysis.py         # Fast parallel metadata analyzer
├── json_file_analysis.py    # Legacy single-threaded analyzer
├── requirements.txt         # Python dependencies
├── schema.sql              # PostgreSQL schema design
└── README.md               # This file
```

## Database Schema

### Tables

**`arxiv_papers`**: Main table storing paper metadata
- Primary key: `id` (arXiv paper ID)
- Full-text search indexes on: authors, title, abstract
- Stores: metadata, PDF paths, version history

**`arxiv_categories`**: Category vocabulary
- Stores all unique arXiv categories

**`arxiv_paper_categories`**: Many-to-many junction table
- Links papers to multiple categories
- Indexed for efficient category queries

### Key Features
- Full-text search with GIN indexes
- Automatic `updated_at` timestamp triggers
- JSONB columns for flexible nested data
- Foreign key constraints with cascading deletes

## Database Package

The `database` package provides clean abstractions for database operations:

```python
from database import DatabaseManager, get_database_url_from_env

# Create manager from environment variables
db_url = get_database_url_from_env()
db_manager = DatabaseManager(db_host='localhost', db_name='arxiv')

# Use session context manager (auto-commit/rollback)
with db_manager.session_scope() as session:
    session.add(paper)
    # Automatically commits on success, rolls back on exception
```

**Features**:
- Connection pooling (configurable pool size)
- Context managers for safe transaction handling
- Environment variable support with sensible defaults
- Lazy session creation

## Data Source

Download the arXiv metadata snapshot from:
- **Kaggle**: https://www.kaggle.com/datasets/Cornell-University/arxiv
- **File**: `arxiv-metadata-oai-snapshot.json` (~5GB)
- **Format**: JSONL (one paper per line)

## Development Roadmap

- [x] Environment configuration template
- [x] Database schema design
- [x] Database connection manager
- [x] Git repository with proper .gitignore
- [ ] SQLAlchemy ORM models
- [ ] Alembic migrations
- [ ] Data loading scripts
- [ ] PDF download manager
- [ ] Testing suite
- [ ] Documentation

## Performance

- **Analysis speed**: Processes 1.7M records using multiprocessing
- **Database**: Optimized with GIN indexes for full-text search
- **Connection pooling**: 10 connections, 20 overflow

## Notes

- Always respect arXiv's rate limits when downloading PDFs
- Large JSON file not included in repository (add to .gitignore)
- PDF files stored separately, paths tracked in database

## License

This project is for educational purposes. See [LICENSE](LICENSE) for details.

## Contributing

This is a learning project. Feel free to fork and experiment!

---

**Disclaimer**: This project is not affiliated with arXiv. Please respect arXiv's [terms of service](https://arxiv.org/help/api/tou) when using their data.
