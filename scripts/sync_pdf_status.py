#!/usr/bin/env python3
"""
Synchronize PDF download status with the filesystem.

Scans the configured PDF directory, checks for each paper whether its PDF
exists, and updates the database `pdf_downloaded` flag accordingly.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Add project root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(override=True)

from database.database import DatabaseManager
from database.models import ArxivPaper  # noqa: E402


class PDFSyncer:
    def __init__(
        self,
        db_url: Optional[str],
        db_host: str,
        db_port: int,
        db_name: str,
        db_user: str,
        db_password: str,
        pdf_base_path: str,
        chunk_size: int = 1000,
    ):
        self.pdf_base_path = Path(pdf_base_path)
        self.chunk_size = chunk_size
        self.db_manager = DatabaseManager(
            db_url=db_url,
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
        )
        self.db_manager.create_engine_and_session()
        engine_url = self.db_manager.engine.url.render_as_string(hide_password=True)
        print(f"✓ Connected to database: {engine_url}")
        print(f"PDF base path: {self.pdf_base_path}")

    def resolve_path(self, pdf_path: Optional[str], paper_id: str) -> Path:
        """Resolve the path to a PDF for a paper."""
        if pdf_path:
            candidate = Path(pdf_path)
            if not candidate.is_absolute():
                candidate = self.pdf_base_path / candidate
        else:
            candidate = self.pdf_base_path / f"{paper_id}.pdf"
        return candidate

    def sync(self):
        """Scan PDFs and update pdf_downloaded flags."""
        total = 0
        updated_true = 0
        updated_false = 0
        last_id: Optional[str] = None

        while True:
            with self.db_manager.session_scope() as session:
                query = session.query(ArxivPaper).order_by(ArxivPaper.id)
                if last_id:
                    query = query.filter(ArxivPaper.id > last_id)

                batch = query.limit(self.chunk_size).all()
                if not batch:
                    break

                for paper in batch:
                    total += 1
                    path = self.resolve_path(paper.pdf_path, paper.id)
                    exists = path.exists()

                    if exists != paper.pdf_downloaded:
                        paper.pdf_downloaded = exists
                        if exists:
                            updated_true += 1
                        else:
                            updated_false += 1

                    last_id = paper.id

                # Commit once per batch
                print(
                    f"Processed: {total:,} | "
                    f"Set true: {updated_true:,} | "
                    f"Set false: {updated_false:,}",
                    end="\r",
                )

        print()
        print("Sync complete")
        print(f"  Total processed: {total:,}")
        print(f"  Set pdf_downloaded -> true : {updated_true:,}")
        print(f"  Set pdf_downloaded -> false: {updated_false:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync PDF download status with filesystem"
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("SQLALCHEMY_URL"),
        help="Full SQLAlchemy DB URL (takes priority over other DB args)",
    )
    parser.add_argument(
        "--db-host",
        default=os.getenv("DB_HOST", ""),
        help="PostgreSQL host (empty for Unix socket)",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=int(os.getenv("DB_PORT", "5432")),
        help="PostgreSQL port",
    )
    parser.add_argument(
        "--db-name",
        default=os.getenv("DB_NAME", "arxiv"),
        help="Database name",
    )
    parser.add_argument(
        "--db-user",
        default=os.getenv("DB_USER", "postgres"),
        help="Database user",
    )
    parser.add_argument(
        "--db-password",
        default=os.getenv("DB_PASSWORD", ""),
        help="Database password",
    )
    parser.add_argument(
        "--pdf-path",
        default=os.getenv("PDF_BASE_PATH", "./arxiv_pdfs"),
        help="Base path for PDF files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Commit/iteration chunk size",
    )

    args = parser.parse_args()

    syncer = PDFSyncer(
        db_url=args.db_url,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        pdf_base_path=args.pdf_path,
        chunk_size=args.chunk_size,
    )
    syncer.sync()


if __name__ == "__main__":
    main()
