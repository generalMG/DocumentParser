# ArXiv Database Project

Python tooling for loading the arXiv metadata snapshot into PostgreSQL, storing PDFs directly in the database, and running a PaddleOCR-VL pipeline (service + client + visualization).

## What you can do
- **Process arXiv papers end-to-end** with a unified script (`scripts/process_arxiv.py`)
  - Load metadata from JSONL into PostgreSQL
  - Download PDFs directly into the database (BYTEA storage)
  - Concurrent downloads with rate limiting
  - Resume from where you left off automatically
- Run a FastAPI PaddleOCR-VL service (GPU or CPU) that strips image payloads and exposes `/ocr` and `/ocr_page`
- Send downloaded PDFs to the OCR service with resumable, page-by-page processing and JSON output (`scripts/ocr_client.py`)
- Visualize OCR layout boxes over the source PDF (`scripts/visualize_ocr.py`)
- Quickly explore the metadata snapshot and plot year/version trends (`fast_analysis.py`)

## Requirements
- Python 3.8+
- PostgreSQL 12+
- ~5GB disk for the metadata JSON
- Database storage for PDFs (expect ~0.37MB average per paper)
- Optional GPU for PaddleOCR-VL (install `paddlepaddle-gpu` instead of `paddlepaddle`)

Install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r requirements.txt
```
`fastapi`, `uvicorn`, `requests`, `PyPDF2`, `reportlab`, and PaddleOCR packages are needed for the OCR pipeline.
### Recommendation
Use `uv package manager` for faster and more reliable installation.

## Configuration
Set environment variables directly or create a `.env` file (loaded by scripts via `python-dotenv`):

| Variable | Description | Default |
|----------|-------------|---------|
| `SQLALCHEMY_URL` | Full DB URL (overrides other DB vars) | `postgresql://user@/arxiv` |
| `DB_HOST` | Postgres host (`""` for Unix socket) | `""` |
| `DB_PORT` | Postgres port | `5432` |
| `DB_NAME` | Database name | `arxiv` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | `""` |
| `ARXIV_DATA_PATH` | Path to the metadata JSONL snapshot | `./arxiv-metadata-oai-snapshot.json` |
| `DOWNLOAD_DELAY` | Seconds to sleep between PDF downloads | `3.0` |

**Note:** PDFs are now stored directly in PostgreSQL as binary data (BYTEA). `PDF_BASE_PATH` is only used by legacy scripts.

Unix socket example:
```
DB_HOST=
SQLALCHEMY_URL=postgresql://your_user@/arxiv
```

TCP example:
```
DB_HOST=localhost
SQLALCHEMY_URL=postgresql://your_user:password@localhost:5432/arxiv
```

## Database setup
```bash
createdb arxiv          # once
alembic upgrade head    # apply migrations (001, 002, 003)
```

Migrations:
- `001`: Initial schema (papers, categories, relationships)
- `002`: PDF download tracking columns
- `003`: PDF binary storage (BYTEA column for storing PDFs in database)

## Unified Processing Pipeline (Recommended)

The `process_arxiv.py` script is the main entry point that handles metadata loading and PDF downloading in one go.

### Starting from scratch (empty database)
```bash
# Process first 5000 papers (metadata + PDFs)
python scripts/process_arxiv.py --limit 5000 --workers 4
```

### Resume from where you left off
```bash
# Continue processing 5000 MORE papers (auto-resumes)
python scripts/process_arxiv.py --limit 5000 --workers 4
```
The script automatically:
- Checks what papers are already in the database
- Skips those papers when reading the JSONL
- Loads only NEW papers
- Downloads PDFs for the new papers

### Backfill missing PDFs
If you have papers in the database but some don't have PDFs:
```bash
# Check how many papers need PDFs
psql -d arxiv -c "SELECT COUNT(*) FROM arxiv_papers WHERE pdf_content IS NULL;"

# Download PDFs for papers that don't have them
python scripts/process_arxiv.py --skip-metadata --limit 3000 --workers 4
```

### Common scenarios

**Scenario 1: You have 4056 papers in DB, only 1294 have PDFs**
```bash
# First, backfill the 2762 missing PDFs
python scripts/process_arxiv.py --skip-metadata --limit 2762 --workers 4

# Then continue loading new papers
python scripts/process_arxiv.py --limit 5000 --workers 4
```

**Scenario 2: You want to load metadata only (no PDF downloads)**
```bash
python scripts/process_arxiv.py --limit 10000 --skip-download
```

**Scenario 3: Process everything with 8 concurrent workers**
```bash
python scripts/process_arxiv.py --limit 10000 --workers 8 --delay 2.0
```

### Pipeline features
- **Automatic resume**: Skips papers already in database (use `--no-resume` to disable)
- **Concurrent downloads**: Multi-threaded PDF downloads (default: 4 workers)
- **Rate limiting**: Respects arXiv terms of service with configurable delays
- **Error handling**: Tracks failed downloads with error messages for retry
- **Progress tracking**: Real-time progress bars for metadata and PDFs
- **Status syncing**: Automatically ensures `pdf_downloaded` flag matches `pdf_content` presence
- **Database storage**: PDFs stored as binary data (BYTEA) directly in PostgreSQL

### Check your progress
```bash
# Total papers
psql -d arxiv -c "SELECT COUNT(*) as total FROM arxiv_papers;"

# Papers with PDFs
psql -d arxiv -c "SELECT COUNT(*) as with_pdfs FROM arxiv_papers WHERE pdf_content IS NOT NULL;"

# Papers needing PDFs
psql -d arxiv -c "SELECT COUNT(*) as need_pdfs FROM arxiv_papers WHERE pdf_content IS NULL;"

# Failed downloads
psql -d arxiv -c "SELECT COUNT(*) as failed FROM arxiv_papers WHERE pdf_download_error IS NOT NULL;"
```

## Legacy/Alternative Scripts

The following individual scripts are still available if you need fine-grained control:

### Load metadata only
```bash
python scripts/load_arxiv_data.py --max-records 1000
```

### Download PDFs to filesystem
```bash
python scripts/download_pdfs.py --limit 200 --delay 3
```

### Load filesystem PDFs into database
```bash
python scripts/load_pdfs_to_db.py --limit 1000
```

### Sync status flags
```bash
python scripts/sync_pdf_status.py
```

## PaddleOCR-VL service
```bash
# GPU example
python scripts/ocr_service.py --host 0.0.0.0 --port 8000 --device gpu:0 --workers 1

# CPU fallback
python scripts/ocr_service.py --device cpu
```
- FastAPI app with a process pool; each worker preloads PaddleOCR-VL and runs a warm-up
- `/ocr` processes whole PDFs (optionally limited by `pages` form field)
- `/ocr_page` processes a single page and returns `total_pages`
- Filters out image blobs for lean JSON; injects DPI info when available
- Monkeypatches `safetensors.safe_open` to read Paddle weights with torch as a backend (handles bfloat16 models)
- Logs GPU memory info when available

## Send PDFs to OCR (resumable client)
```bash
python scripts/ocr_client.py \
  --service-url http://localhost:8000/ocr_page \
  --limit 10 \
  --pages 5 \           # optional: cap per-PDF page count
  --output-dir outputs/ocr_results \
  --offset 0
```
- Pulls rows where `pdf_downloaded=true` and `pdf_path` is set
- Processes page-by-page, writes `paper_id.json`, and maintains `.checkpoint` files to resume partial runs
- Keeps incremental results sorted even after restarts

## Visualize OCR output
```bash
python scripts/visualize_ocr.py \
  --json outputs/ocr_results/0704.0001.json \
  --pdf /path/to/0704.0001.pdf \
  --page 0 \
  --show-content \
  --output outputs/visualizations/0704.0001_p0.png
```
Renders PaddleOCR layout boxes on top of the PDF page and (optionally) a side panel of extracted text.

## Quick metadata analysis
```bash
python fast_analysis.py ./arxiv-metadata-oai-snapshot.json --cpu_count 8
```
Runs a multiprocessing scan of the JSONL file and saves `outputs/arxiv_analysis_fast.png` with year/version histograms.

## Project layout
```
alembic/
  env.py
  versions/
    001_initial_schema.py
    002_add_pdf_download_tracking.py
    003_add_pdf_binary_storage.py
database/
  database.py          # URL builder, engine/session manager
  models.py            # ORM models + indexes (includes pdf_content BYTEA)
scripts/
  process_arxiv.py     # Unified pipeline: metadata + PDF downloads (RECOMMENDED)
  load_arxiv_data.py   # Legacy: JSONL -> Postgres loader
  download_pdfs.py     # Legacy: Download PDFs to filesystem
  load_pdfs_to_db.py   # Legacy: Load filesystem PDFs into database
  sync_pdf_status.py   # Legacy: Reconcile pdf_downloaded flags
  ocr_service.py       # FastAPI PaddleOCR-VL service
  ocr_client.py        # Resumable page-by-page OCR client
  visualize_ocr.py     # Draw OCR boxes on PDF pages
fast_analysis.py       # Parallel metadata explorer
schema.sql             # Reference DDL (mirrors Alembic head)
requirements.txt
README.md
```

## Notes & pitfalls
- **Large files**: The JSON snapshot is ~5GB; PDFs stored in PostgreSQL will grow your database significantly (expect ~0.37MB average per paper)
- **Database storage**: With BYTEA storage, 10k papers = ~3.7GB database growth. Plan your PostgreSQL storage accordingly
- **Socket vs TCP**: Empty `DB_HOST` builds a Unix-socket URL; set `DB_HOST=localhost` if peer auth fails
- **Respect arXiv rate limits**: Default delay is 3 seconds between downloads. Do not decrease below 2 seconds
- **Resume behavior**: The unified script automatically resumes from where it left off. To reload existing papers, use `--no-resume`
- **Concurrent downloads**: More workers = faster processing, but respect rate limits. Recommended: 4-8 workers
- **OCR models**: The OCR service expects models under `~/.paddlex/official_models` by default; first run will download them

## Storage comparison
- **Filesystem**: ~3.7GB for 10k PDFs on disk
- **Database (BYTEA)**: ~3.7GB added to PostgreSQL database
- **Advantage of DB storage**: Single source of truth, easier backups, no file path management, transactional consistency

## License
Educational use only. See `LICENSE`.
