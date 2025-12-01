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
from pathlib import Path
from typing import List, Tuple

import requests
from dotenv import load_dotenv

from database.database import DatabaseManager
from database.models import ArxivPaper

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


def main():
    parser = argparse.ArgumentParser(description="Send downloaded PDFs to OCR service")
    parser.add_argument("--service-url", default="http://localhost:8000/ocr", help="OCR service endpoint")
    parser.add_argument("--output-dir", default="outputs/ocr_results", help="Directory to save JSON outputs")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to process")
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

        with pdf_file.open("rb") as f:
            files = {"file": (pdf_file.name, f, "application/pdf")}
            data = {"pages": args.pages}
            try:
                resp = requests.post(args.service_url, files=files, data=data, timeout=300)
            except Exception as e:
                print(f"{paper_id}: request failed: {e}")
                continue

        if resp.status_code != 200:
            print(f"{paper_id}: service error {resp.status_code} -> {resp.text}")
            continue

        try:
            payload = resp.json()
        except Exception as e:
            print(f"{paper_id}: failed to parse JSON: {e}")
            continue

        out_path = output_dir / f"{paper_id}.json"
        with out_path.open("w", encoding="utf-8") as out_f:
            json.dump(payload, out_f, ensure_ascii=False, indent=2)
        print(f"{paper_id}: saved {out_path}")


if __name__ == "__main__":
    main()
