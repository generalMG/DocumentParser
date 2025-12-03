#!/usr/bin/env python3
"""
Client script to fetch PDFs from the database and send them to the OCR service.

- Pulls rows with pdf_downloaded = true.
- Sends each PDF to the FastAPI OCR service.
- Saves JSON responses to an output directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.database import DatabaseManager  # noqa: E402
from database.models import ArxivPaper  # noqa: E402

load_dotenv(override=True)


def fetch_pdfs(db: DatabaseManager, limit: int, offset: int) -> List[Tuple[str, str]]:
    with db.session_scope() as session:
        query = session.query(ArxivPaper).filter(
            ArxivPaper.pdf_downloaded.is_(True),
            ArxivPaper.pdf_path.isnot(None),
        ).order_by(ArxivPaper.id)
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        rows = query.all()
        return [(row.id, row.pdf_path) for row in rows if row.pdf_path]


def process_pdf_incrementally(
    paper_id: str,
    pdf_file: Path,
    service_url: str,
    output_dir: Path,
    max_pages: int = None,
) -> bool:
    """Process a single PDF page by page with checkpoint support."""
    checkpoint_path = output_dir / f"{paper_id}.checkpoint"
    out_path = output_dir / f"{paper_id}.json"

    # Load checkpoint if exists
    completed_pages = set()
    results_dict = {}
    total_pages = None

    if checkpoint_path.exists():
        try:
            with checkpoint_path.open("r") as f:
                checkpoint = json.load(f)
                completed_pages = set(checkpoint.get("completed_pages", []))
                total_pages = checkpoint.get("total_pages")
                print(f"{paper_id}: resuming from checkpoint (completed: {len(completed_pages)} pages)")
        except Exception as e:
            print(f"{paper_id}: failed to load checkpoint: {e}, starting fresh")

    # Load existing results if available
    if out_path.exists():
        try:
            with out_path.open("r") as f:
                existing_data = json.load(f)
                results_dict = {r.get("page", i): r for i, r in enumerate(existing_data.get("results", []))}
        except Exception as e:
            print(f"{paper_id}: failed to load existing results: {e}")

    # Read PDF file once
    try:
        with pdf_file.open("rb") as f:
            pdf_bytes = f.read()
    except Exception as e:
        print(f"{paper_id}: failed to read PDF: {e}")
        return False

    # Determine total pages (from checkpoint or first request)
    page_num = 0
    while total_pages is None or (max_pages and page_num < max_pages) or (not max_pages and page_num < total_pages):
        if page_num in completed_pages:
            page_num += 1
            continue

        # Request single page
        try:
            files = {"file": (pdf_file.name, pdf_bytes, "application/pdf")}
            data = {"page": page_num}
            resp = requests.post(service_url, files=files, data=data, timeout=6000)
        except Exception as e:
            print(f"{paper_id}: page {page_num} request failed: {e}")
            return False

        if resp.status_code != 200:
            print(f"{paper_id}: page {page_num} service error {resp.status_code} -> {resp.text}")
            return False

        try:
            payload = resp.json()
        except Exception as e:
            print(f"{paper_id}: page {page_num} failed to parse JSON: {e}")
            return False

        # Extract total pages from first response
        if total_pages is None:
            total_pages = payload.get("total_pages")
            if total_pages is None:
                print(f"{paper_id}: could not determine total pages")
                return False
            print(f"{paper_id}: total pages = {total_pages}")

        # Store page result
        page_result = payload.get("result")
        if page_result is not None:
            results_dict[page_num] = page_result
            completed_pages.add(page_num)

            # Save incrementally
            sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]
            try:
                with out_path.open("w", encoding="utf-8") as out_f:
                    json.dump({"results": sorted_results}, out_f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"{paper_id}: failed to save results: {e}")
                return False

            # Update checkpoint
            try:
                with checkpoint_path.open("w", encoding="utf-8") as ckpt_f:
                    json.dump({
                        "completed_pages": sorted(list(completed_pages)),
                        "total_pages": total_pages,
                    }, ckpt_f, indent=2)
            except Exception as e:
                print(f"{paper_id}: failed to save checkpoint: {e}")

            print(f"{paper_id}: page {page_num}/{total_pages} completed")

        page_num += 1

        # Check if we've processed enough pages
        if max_pages and page_num >= max_pages:
            break
        if page_num >= total_pages:
            break

    # Clean up checkpoint when done
    if len(completed_pages) >= total_pages or (max_pages and len(completed_pages) >= max_pages):
        try:
            checkpoint_path.unlink(missing_ok=True)
            print(f"{paper_id}: processing complete, checkpoint removed")
        except Exception as e:
            print(f"{paper_id}: failed to remove checkpoint: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Send downloaded PDFs to OCR service")
    parser.add_argument("--service-url", default="http://localhost:8000/ocr_page", help="OCR service endpoint")
    parser.add_argument("--output-dir", default="outputs/ocr_results", help="Directory to save JSON outputs")
    parser.add_argument("--pages", type=int, default=None, help="Number of pages to process (default: all)")
    parser.add_argument("--limit", type=int, default=10, help="Max number of PDFs to send")
    parser.add_argument("--offset", type=int, default=0, help="Offset for DB query")
    parser.add_argument("--db-url", default=os.getenv("SQLALCHEMY_URL"), help="Optional DB URL override")
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", ""), help="DB host")
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")), help="DB port")
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "arxiv"), help="DB name")
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"), help="DB user")
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD", ""), help="DB password")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = DatabaseManager(
        db_url=args.db_url,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
    )
    db.create_engine_and_session()

    rows = fetch_pdfs(db, limit=args.limit, offset=args.offset)
    print(f"Found {len(rows)} PDFs to process")

    for paper_id, pdf_path in rows:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            print(f"Skipping missing file: {pdf_path}")
            continue

        success = process_pdf_incrementally(
            paper_id=paper_id,
            pdf_file=pdf_file,
            service_url=args.service_url,
            output_dir=output_dir,
            max_pages=args.pages,
        )

        if success:
            print(f"{paper_id}: successfully processed")
        else:
            print(f"{paper_id}: processing failed")


if __name__ == "__main__":
    main()
