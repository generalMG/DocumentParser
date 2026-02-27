#!/usr/bin/env python3
"""
Download PDFs from arXiv.org with rate limiting and retry logic.

- Selects papers where `pdf_downloaded` is false.
- Respects arXiv rate limits via a configurable delay.
- Supports category filtering.
- Tracks errors in `pdf_download_error` and timestamps attempts.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

# Add project root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(override=True)

from database.database import DatabaseManager  # noqa: E402
from database.models import ArxivPaper  # noqa: E402
from database.path_security import normalize_arxiv_id, safe_pdf_path  # noqa: E402


class PDFDownloader:
    def __init__(
        self,
        db_url: Optional[str],
        db_host: str,
        db_port: int,
        db_name: str,
        db_user: str,
        db_password: str,
        pdf_base_path: str,
        delay: float = 3.0,
        retries: int = 3,
        timeout: float = 30.0,
        max_pdf_size_mb: float = 100.0,
    ):
        self.delay = delay
        self.retries = retries
        self.timeout = timeout
        self.max_pdf_size_bytes = int(max_pdf_size_mb * 1024 * 1024)
        self.pdf_base_path = Path(pdf_base_path)
        self.pdf_base_path.mkdir(parents=True, exist_ok=True)

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
        print(f"Max PDF size: {max_pdf_size_mb:.1f} MB")

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "arxiv-pdf-downloader/1.0"})

    @staticmethod
    def _is_pdf_content_type(content_type: Optional[str]) -> bool:
        if not content_type:
            return True
        lowered = content_type.lower()
        return "pdf" in lowered or "octet-stream" in lowered

    def build_pdf_url(self, arxiv_id: str) -> str:
        normalized_id = normalize_arxiv_id(arxiv_id)
        return urljoin("https://arxiv.org/", f"pdf/{normalized_id}.pdf")

    def download_pdf(self, arxiv_id: str, dest_path: Path) -> Optional[str]:
        url = self.build_pdf_url(arxiv_id)
        last_error = None
        for attempt in range(1, self.retries + 1):
            tmp_path = dest_path.with_name(f"{dest_path.name}.part")
            try:
                with self.session.get(url, timeout=self.timeout, stream=True) as resp:
                    if resp.status_code != 200:
                        last_error = f"HTTP {resp.status_code}"
                        continue

                    content_type = resp.headers.get("Content-Type")
                    if not self._is_pdf_content_type(content_type):
                        last_error = f"Unexpected content type: {content_type}"
                        continue

                    content_length = resp.headers.get("Content-Length")
                    if content_length:
                        try:
                            if int(content_length) > self.max_pdf_size_bytes:
                                raise ValueError(
                                    f"PDF exceeds size limit ({content_length} > {self.max_pdf_size_bytes} bytes)"
                                )
                        except ValueError as exc:
                            if "PDF exceeds size limit" in str(exc):
                                raise

                    total_bytes = 0
                    signature = b""
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if not chunk:
                                continue

                            total_bytes += len(chunk)
                            if total_bytes > self.max_pdf_size_bytes:
                                raise ValueError(
                                    f"PDF exceeds size limit ({self.max_pdf_size_bytes} bytes)"
                                )

                            if len(signature) < 5:
                                need = 5 - len(signature)
                                signature += chunk[:need]

                            f.write(chunk)

                    if not signature.startswith(b"%PDF-"):
                        raise ValueError("Downloaded payload is not a PDF")

                    tmp_path.replace(dest_path)
                    return None
            except Exception as e:
                last_error = str(e)
                if tmp_path.exists():
                    tmp_path.unlink()

            if attempt < self.retries:
                time.sleep(self.delay)
        return last_error

    def fetch_candidates(
        self, session, limit: int, categories: Optional[List[str]], include_errors: bool
    ) -> list[str]:
        """Return a list of paper IDs to download."""
        q = session.query(ArxivPaper.id).filter(
            ArxivPaper.pdf_downloaded.is_(False)
        )
        if not include_errors:
            q = q.filter(ArxivPaper.pdf_download_error.is_(None))
        if categories:
            pattern = "%{}%"
            ors = []
            for cat in categories:
                ors.append(ArxivPaper.categories.ilike(pattern.format(cat)))
            if ors:
                from sqlalchemy import or_
                q = q.filter(or_(*ors))
        return [row[0] for row in q.order_by(ArxivPaper.id).limit(limit).all()]

    def run(self, limit: int, auto_commit: bool, include_errors: bool):
        total = 0
        success = 0
        failures = 0

        with self.db_manager.session_scope() as session:
            candidates = self.fetch_candidates(
                session, limit=limit, categories=self.categories, include_errors=include_errors
            )

        print(f"Found {len(candidates)} candidates to download")
        for idx, paper_id in enumerate(candidates, 1):
            started_at = datetime.utcnow()
            try:
                normalized_id = normalize_arxiv_id(paper_id)
                dest = safe_pdf_path(self.pdf_base_path, normalized_id)
            except ValueError as exc:
                normalized_id = None
                err = f"Unsafe arXiv ID for filesystem path: {exc}"
            else:
                err = self.download_pdf(normalized_id, dest)

            with self.db_manager.session_scope() as session:
                db_paper = session.get(ArxivPaper, paper_id)
                if db_paper:
                    db_paper.pdf_download_attempted_at = started_at
                    if err is None:
                        db_paper.pdf_downloaded = True
                        db_paper.pdf_download_error = None
                        success += 1
                    else:
                        db_paper.pdf_downloaded = False
                        db_paper.pdf_download_error = err
                        failures += 1
                total += 1

            print(
                f"[{idx}/{len(candidates)}] {paper_id} -> "
                f"{'ok' if err is None else 'fail: ' + err}"
            )

            if idx < len(candidates) and auto_commit:
                time.sleep(self.delay)

        print("Download summary:")
        print(f"  Total attempted: {total}")
        print(f"  Success: {success}")
        print(f"  Failures: {failures}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download arXiv PDFs with rate limiting and retry logic"
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
        "--delay",
        type=float,
        default=float(os.getenv("DOWNLOAD_DELAY", "3.0")),
        help="Delay between downloads (seconds)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts per file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout (seconds)",
    )
    parser.add_argument(
        "--max-pdf-size-mb",
        type=float,
        default=float(os.getenv("MAX_PDF_SIZE_MB", "100")),
        help="Maximum PDF size to download per paper in MB (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of PDFs to download",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional list of category substrings to filter (e.g., cs.AI math.)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Enable auto mode (sleep between downloads)",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Include papers that previously failed (pdf_download_error is not NULL)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    downloader = PDFDownloader(
        db_url=args.db_url,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        pdf_base_path=args.pdf_path,
        delay=args.delay,
        retries=args.retries,
        timeout=args.timeout,
        max_pdf_size_mb=args.max_pdf_size_mb,
    )
    downloader.categories = args.categories
    downloader.run(limit=args.limit, auto_commit=args.auto, include_errors=args.retry_errors)


if __name__ == "__main__":
    main()
