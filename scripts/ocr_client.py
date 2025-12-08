#!/usr/bin/env python3
"""
OCR client that fetches PDFs from database and stores results back to database.

- Pulls papers with pdf_content (PDF stored in DB) that haven't been OCR processed yet
- Sends each PDF to the FastAPI OCR service
- Saves OCR results directly to the database (ocr_results JSONB column)
- Tracks processing status via ocr_processed, ocr_processed_at, ocr_error columns
"""

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.database import DatabaseManager  # noqa: E402
from database.models import ArxivPaper  # noqa: E402

load_dotenv(override=True)


def fetch_papers_for_ocr(db: DatabaseManager, limit: int, offset: int) -> List[str]:
    """Fetch paper IDs that have PDF content but haven't been OCR processed."""
    with db.session_scope() as session:
        query = session.query(ArxivPaper.id).filter(
            ArxivPaper.pdf_content.isnot(None),
            ArxivPaper.ocr_processed.is_(False),
        ).order_by(ArxivPaper.id)

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
    error: str = None
):
    """Save OCR results to database."""
    with db.session_scope() as session:
        paper = session.query(ArxivPaper).filter(ArxivPaper.id == paper_id).first()
        if paper:
            paper.ocr_results = results
            if completed:
                paper.ocr_processed = True
                paper.ocr_processed_at = datetime.utcnow()
            if error:
                paper.ocr_error = error
            session.commit()


def process_pdf_to_db(
    db: DatabaseManager,
    paper_id: str,
    pdf_bytes: bytes,
    service_url: str,
    max_pages: int = None,
) -> bool:
    """Process a single PDF page by page and save results to database."""

    # Load existing partial results for resume support
    existing_data = get_partial_results(db, paper_id)
    results_dict = {}
    completed_pages = set()
    total_pages = existing_data.get("total_pages")

    if existing_data.get("results"):
        for i, r in enumerate(existing_data["results"]):
            page_num = r.get("page", i)
            results_dict[page_num] = r
            completed_pages.add(page_num)
        if completed_pages:
            print(f"{paper_id}: resuming from checkpoint (completed: {len(completed_pages)} pages)")

    page_num = 0
    while True:
        # Skip already completed pages
        if page_num in completed_pages:
            page_num += 1
            # Check termination conditions
            if max_pages and page_num >= max_pages:
                break
            if total_pages and page_num >= total_pages:
                break
            continue

        # Send request to OCR service
        try:
            files = {"file": (f"{paper_id}.pdf", pdf_bytes, "application/pdf")}
            data = {"page": page_num}
            resp = requests.post(service_url, files=files, data=data, timeout=6000)
        except Exception as e:
            error_msg = f"page {page_num} request failed: {e}"
            print(f"{paper_id}: {error_msg}")
            save_ocr_results(db, paper_id, _build_results(results_dict, total_pages), error=error_msg)
            return False

        if resp.status_code != 200:
            error_msg = f"page {page_num} service error {resp.status_code}: {resp.text}"
            print(f"{paper_id}: {error_msg}")
            save_ocr_results(db, paper_id, _build_results(results_dict, total_pages), error=error_msg)
            return False

        try:
            payload = resp.json()
        except Exception as e:
            error_msg = f"page {page_num} failed to parse JSON: {e}"
            print(f"{paper_id}: {error_msg}")
            save_ocr_results(db, paper_id, _build_results(results_dict, total_pages), error=error_msg)
            return False

        # Extract total pages from first response
        if total_pages is None:
            total_pages = payload.get("total_pages")
            if total_pages is None:
                error_msg = "could not determine total pages"
                print(f"{paper_id}: {error_msg}")
                save_ocr_results(db, paper_id, {}, error=error_msg)
                return False
            print(f"{paper_id}: total pages = {total_pages}")

        # Store page result
        page_result = payload.get("result")
        if page_result is not None:
            results_dict[page_num] = page_result
            completed_pages.add(page_num)

            # Save incrementally to database
            save_ocr_results(db, paper_id, _build_results(results_dict, total_pages))
            print(f"{paper_id}: page {page_num + 1}/{total_pages} completed")

        page_num += 1

        # Check termination conditions
        if max_pages and page_num >= max_pages:
            break
        if page_num >= total_pages:
            break

    # Mark as completed
    is_complete = len(completed_pages) >= total_pages or (max_pages and len(completed_pages) >= max_pages)
    if is_complete:
        save_ocr_results(db, paper_id, _build_results(results_dict, total_pages), completed=True)
        print(f"{paper_id}: processing complete")

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
    parser.add_argument("--db-url", default=os.getenv("SQLALCHEMY_URL"), help="Optional DB URL override")
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", ""), help="DB host")
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")), help="DB port")
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "arxiv"), help="DB name")
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"), help="DB user")
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD", ""), help="DB password")
    args = parser.parse_args()

    db = DatabaseManager(
        db_url=args.db_url,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
    )
    db.create_engine_and_session()

    paper_ids = fetch_papers_for_ocr(db, limit=args.limit, offset=args.offset)
    print(f"Found {len(paper_ids)} PDFs to process")

    success_count = 0
    fail_count = 0

    for paper_id in paper_ids:
        print(f"\n{'='*50}")
        print(f"Processing: {paper_id}")
        print('='*50)

        pdf_bytes = get_pdf_content(db, paper_id)
        if not pdf_bytes:
            print(f"{paper_id}: no PDF content in database, skipping")
            fail_count += 1
            continue

        success = process_pdf_to_db(
            db=db,
            paper_id=paper_id,
            pdf_bytes=pdf_bytes,
            service_url=args.service_url,
            max_pages=args.pages,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*50}")
    print(f"Processing complete: {success_count} succeeded, {fail_count} failed")
    print('='*50)


if __name__ == "__main__":
    main()
