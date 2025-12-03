# ArXiv Database Project

Python tooling for loading the arXiv metadata snapshot into PostgreSQL, downloading PDFs with status tracking, and running a PaddleOCR-VL pipeline (service + client + visualization).

## What you can do
- Load 1.7M+ arXiv JSONL records with upserts and category linking (`scripts/load_arxiv_data.py`)
- Track and download PDFs with retries, rate limiting, and error logging (`scripts/download_pdfs.py`)
- Reconcile DB download flags with what is actually on disk (`scripts/sync_pdf_status.py`)
- Run a FastAPI PaddleOCR-VL service (GPU or CPU) that strips image payloads and exposes `/ocr` and `/ocr_page`
- Send downloaded PDFs to the OCR service with resumable, page-by-page processing and JSON output (`scripts/ocr_client.py`)
- Visualize OCR layout boxes over the source PDF (`scripts/visualize_ocr.py`)
- Quickly explore the metadata snapshot and plot year/version trends (`fast_analysis.py`)

## Requirements
- Python 3.8+
- PostgreSQL 12+
- ~5GB disk for the metadata JSON; additional space for PDFs and OCR outputs
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
| `PDF_BASE_PATH` | Directory to store PDFs | `./arxiv_pdfs` |
| `ARXIV_DATA_PATH` | Path to the metadata JSONL snapshot | `./arxiv-metadata-oai-snapshot.json` |
| `DOWNLOAD_DELAY` | Seconds to sleep between PDF downloads | `3.0` |

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
alembic upgrade head    # apply migrations 001 + 002
# (or psql -d arxiv -f schema.sql if you are avoiding Alembic)
```

## Load metadata into Postgres
```bash
# Quick smoke test
python scripts/load_arxiv_data.py --max-records 1000

# Full load with custom paths
python scripts/load_arxiv_data.py \
  --json-file "$ARXIV_DATA_PATH" \
  --pdf-path "$PDF_BASE_PATH" \
  --batch-size 2000
```
Highlights:
- Upserts on `id` (keeps existing `pdf_downloaded` flags and errors intact)
- Generates `pdf_path` for each record using `PDF_BASE_PATH`
- Rebuilds category links each batch; inserts new categories on the fly
- Progress logging with records/sec

## Download PDFs
```bash
python scripts/download_pdfs.py \
  --limit 200 \
  --pdf-path "$PDF_BASE_PATH" \
  --delay 3 \
  --auto \
  --categories cs.AI math. \
  --retry-errors        # optional: include rows with past failures
```
- Picks `pdf_downloaded = false` rows (skips past errors unless `--retry-errors`)
- Respects `DOWNLOAD_DELAY`, retries HTTP up to `--retries`, and writes timestamps/errors
- Simple category substring filter to throttle by subject area

## Sync DB flags to the filesystem
```bash
python scripts/sync_pdf_status.py --pdf-path "$PDF_BASE_PATH" --chunk-size 2000
```
Scans the PDF directory and flips `pdf_downloaded` true/false in batches so the DB reflects what is actually on disk.

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
database/
  database.py          # URL builder, engine/session manager
  models.py            # ORM models + indexes
scripts/
  load_arxiv_data.py   # JSONL -> Postgres loader (upsert + categories)
  download_pdfs.py     # Download PDFs with retry/rate limiting
  sync_pdf_status.py   # Reconcile pdf_downloaded with filesystem
  ocr_service.py       # FastAPI PaddleOCR-VL service
  ocr_client.py        # Resumable page-by-page OCR client
  visualize_ocr.py     # Draw OCR boxes on PDF pages
fast_analysis.py       # Parallel metadata explorer
schema.sql             # Reference DDL (mirrors Alembic head)
requirements.txt
README.md
```

## Notes & pitfalls
- Large files: the JSON snapshot is ~5GB; PDFs will grow storage quickly
- Socket vs TCP: empty `DB_HOST` builds a Unix-socket URL; set `DB_HOST=localhost` if peer auth fails
- Respect arXiv rate limits when downloading PDFs
- The OCR service expects models under `~/.paddlex/official_models` by default; first run will download them

## License
Educational use only. See `LICENSE`.
