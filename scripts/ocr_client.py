#!/usr/bin/env python3
"""
OCR client that fetches PDFs from database and stores results back to database.

- Pulls papers with pdf_content (PDF stored in DB) that haven't been OCR processed yet
- Sends each PDF to the FastAPI OCR service
- Saves OCR results directly to the database (ocr_results JSONB column)
- Tracks processing status via ocr_processed, ocr_processed_at, ocr_error columns
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.database import DatabaseManager  # noqa: E402
from database.models import ArxivPaper  # noqa: E402

load_dotenv(override=True)


def get_service_info(service_url: str) -> dict:
    """Query OCR service for configuration info including worker count."""
    try:
        # Convert /ocr_page URL to /info URL
        base_url = service_url.rsplit('/', 1)[0]
        info_url = f"{base_url}/info"
        resp = requests.get(info_url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Warning: Could not query service info: {e}")
    return {}


def fetch_papers_for_ocr(
    db: DatabaseManager,
    limit: int,
    offset: int,
    retry_errors: bool = False
) -> List[str]:
    """Fetch paper IDs that have PDF content but haven't been OCR processed.

    Args:
        db: Database manager
        limit: Max papers to fetch
        offset: Offset for pagination
        retry_errors: If True, also include papers with ocr_error set (to retry failed ones)
    """
    with db.session_scope() as session:
        query = session.query(ArxivPaper.id).filter(
            ArxivPaper.pdf_content.isnot(None),
            ArxivPaper.ocr_processed.is_(False),
        )

        # Optionally exclude papers with errors (they failed previously)
        if not retry_errors:
            query = query.filter(ArxivPaper.ocr_error.is_(None))

        query = query.order_by(ArxivPaper.id)

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        return [row.id for row in query.all()]


def get_pdf_content(db: DatabaseManager, paper_id: str) -> Optional[bytes]:
    """Fetch PDF binary content from database."""
    with db.session_scope() as session:
        paper = session.query(ArxivPaper).filter(ArxivPaper.id == paper_id).first()
        if paper and paper.pdf_content:
            return paper.pdf_content
        return None


def get_partial_results(db: DatabaseManager, paper_id: str) -> dict:
    """Get existing partial OCR results for resume support."""
    with db.session_scope() as session:
        paper = session.query(ArxivPaper).filter(ArxivPaper.id == paper_id).first()
        if paper and paper.ocr_results:
            return paper.ocr_results
        return {}


def save_ocr_results(
    db: DatabaseManager,
    paper_id: str,
    results: dict,
    completed: bool = False,
    error: str = None,
    clear_error: bool = False
):
    """Save OCR results to database."""
    with db.session_scope() as session:
        paper = session.query(ArxivPaper).filter(ArxivPaper.id == paper_id).first()
        if paper:
            paper.ocr_results = results
            if completed:
                paper.ocr_processed = True
                paper.ocr_processed_at = datetime.utcnow()
                paper.ocr_error = None  # Clear error on success
            elif error:
                paper.ocr_error = error
            elif clear_error:
                paper.ocr_error = None
            session.commit()


def _extract_page_index(result: dict, fallback_index: int) -> int:
    """Extract page index from OCR result, handling both 'page_index' (string) and 'page' (int)."""
    # OCR service stores as 'page_index' (string)
    if "page_index" in result:
        try:
            return int(result["page_index"])
        except (ValueError, TypeError):
            pass
    # Fallback to 'page' field
    if "page" in result:
        try:
            return int(result["page"])
        except (ValueError, TypeError):
            pass
    # Use fallback index
    return fallback_index


def _process_single_page(
    paper_id: str,
    pdf_bytes: bytes,
    page_num: int,
    service_url: str,
) -> Tuple[int, Optional[dict], Optional[str], Optional[int]]:
    """Process a single page. Returns (page_num, result, error, total_pages)."""
    try:
        files = {"file": (f"{paper_id}.pdf", pdf_bytes, "application/pdf")}
        data = {"page": page_num}
        resp = requests.post(service_url, files=files, data=data, timeout=6000)
    except Exception as e:
        return (page_num, None, f"request failed: {e}", None)

    if resp.status_code != 200:
        return (page_num, None, f"service error {resp.status_code}: {resp.text}", None)

    try:
        payload = resp.json()
    except Exception as e:
        return (page_num, None, f"failed to parse JSON: {e}", None)

    total_pages = payload.get("total_pages")
    page_result = payload.get("result")

    return (page_num, page_result, None, total_pages)


def process_pdf_to_db(
    db: DatabaseManager,
    paper_id: str,
    pdf_bytes: bytes,
    service_url: str,
    max_pages: int = None,
    gpu_workers: int = 1,
) -> bool:
    """Process a single PDF with concurrent page processing."""

    # Load existing partial results for resume support
    existing_data = get_partial_results(db, paper_id)
    results_dict = {}
    completed_pages = set()
    total_pages = existing_data.get("total_pages")

    # Restore previously processed pages
    if existing_data.get("results"):
        for i, r in enumerate(existing_data["results"]):
            page_num = _extract_page_index(r, i)
            results_dict[page_num] = r
            completed_pages.add(page_num)
        if completed_pages:
            print(f"  Resuming: {len(completed_pages)} pages already completed")
            save_ocr_results(db, paper_id, existing_data, clear_error=True)

    # First, get total pages if we don't know it yet
    if total_pages is None:
        page_num, result, error, total_pages = _process_single_page(
            paper_id, pdf_bytes, 0, service_url
        )
        if total_pages is None:
            error_msg = error or "could not determine total pages"
            print(f"  Error: {error_msg}")
            save_ocr_results(db, paper_id, {}, error=error_msg)
            return False

        if result is not None:
            results_dict[0] = result
            completed_pages.add(0)

        print(f"  Total pages: {total_pages}")

    # Determine which pages still need processing
    target_pages = min(total_pages, max_pages) if max_pages else total_pages
    pages_to_process = [p for p in range(target_pages) if p not in completed_pages]

    if not pages_to_process:
        print(f"  All pages already completed")
        save_ocr_results(db, paper_id, _build_results(results_dict, total_pages), completed=True)
        return True

    print(f"  Processing {len(pages_to_process)} pages with {gpu_workers} concurrent worker(s)...")

    # Process pages concurrently
    with ThreadPoolExecutor(max_workers=gpu_workers) as executor:
        futures = {
            executor.submit(
                _process_single_page,
                paper_id,
                pdf_bytes,
                page_num,
                service_url,
            ): page_num
            for page_num in pages_to_process
        }

        processed = 0
        errors = []

        for future in as_completed(futures):
            page_num = futures[future]
            try:
                pg, result, error, _ = future.result()

                if error:
                    errors.append(f"page {pg}: {error}")
                    print(f"  Page {pg + 1}/{total_pages}: ERROR - {error}")
                elif result is not None:
                    results_dict[pg] = result
                    completed_pages.add(pg)
                    processed += 1
                    print(f"  Page {pg + 1}/{total_pages}: OK ({processed}/{len(pages_to_process)})")

            except Exception as e:
                errors.append(f"page {page_num}: exception {e}")
                print(f"  Page {page_num + 1}/{total_pages}: EXCEPTION - {e}")

        # Save results after all pages complete
        if errors:
            error_msg = "; ".join(errors[:3])  # First 3 errors
            if len(errors) > 3:
                error_msg += f" (+{len(errors) - 3} more)"
            save_ocr_results(db, paper_id, _build_results(results_dict, total_pages), error=error_msg)
            print(f"  Completed with {len(errors)} error(s)")
            return False
        else:
            save_ocr_results(db, paper_id, _build_results(results_dict, total_pages), completed=True)
            print(f"  Completed successfully")
            return True


def _build_results(results_dict: dict, total_pages: int = None) -> dict:
    """Build the results JSON structure."""
    sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]
    output = {"results": sorted_results}
    if total_pages:
        output["total_pages"] = total_pages
    output["processed_pages"] = len(sorted_results)
    return output


def main():
    parser = argparse.ArgumentParser(description="Process PDFs from database through OCR service")
    parser.add_argument("--service-url", default="http://localhost:8000/ocr_page", help="OCR service endpoint")
    parser.add_argument("--pages", type=int, default=None, help="Max pages per PDF (default: all)")
    parser.add_argument("--limit", type=int, default=10, help="Max number of PDFs to process")
    parser.add_argument("--offset", type=int, default=0, help="Offset for DB query")
    parser.add_argument("--gpu-workers", type=int, default=None,
                        help="Number of concurrent pages to process (default: auto-detect from OCR service)")
    parser.add_argument("--retry-errors", action="store_true", help="Retry papers that previously failed with errors")
    parser.add_argument("--db-url", default=os.getenv("SQLALCHEMY_URL"), help="Optional DB URL override")
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", ""), help="DB host")
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")), help="DB port")
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "arxiv"), help="DB name")
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"), help="DB user")
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD", ""), help="DB password")
    args = parser.parse_args()

    # Auto-detect GPU workers from OCR service if not specified
    if args.gpu_workers is None:
        service_info = get_service_info(args.service_url)
        gpu_workers = service_info.get("workers", 1)
        print(f"Auto-detected {gpu_workers} GPU worker(s) from OCR service")
    else:
        gpu_workers = args.gpu_workers
        print(f"Using {gpu_workers} GPU worker(s) (user specified)")

    db = DatabaseManager(
        db_url=args.db_url,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
    )
    db.create_engine_and_session()

    paper_ids = fetch_papers_for_ocr(
        db,
        limit=args.limit,
        offset=args.offset,
        retry_errors=args.retry_errors
    )
    print(f"Found {len(paper_ids)} PDFs to process")
    if args.retry_errors:
        print("(including papers with previous errors)")

    if not paper_ids:
        print("No papers to process.")
        return

    success_count = 0
    fail_count = 0

    # Process papers sequentially, but pages concurrently
    for i, paper_id in enumerate(paper_ids, 1):
        print(f"\n[{i}/{len(paper_ids)}] {paper_id}")
        print("-" * 50)

        pdf_bytes = get_pdf_content(db, paper_id)
        if not pdf_bytes:
            print(f"  No PDF content in database, skipping")
            fail_count += 1
            continue

        success = process_pdf_to_db(
            db=db,
            paper_id=paper_id,
            pdf_bytes=pdf_bytes,
            service_url=args.service_url,
            max_pages=args.pages,
            gpu_workers=gpu_workers,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*50}")
    print(f"Done: {success_count} succeeded, {fail_count} failed")
    print('='*50)


if __name__ == "__main__":
    main()
