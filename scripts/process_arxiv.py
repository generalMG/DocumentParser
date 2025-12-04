#!/usr/bin/env python3
"""
Unified ArXiv Processing Script

This script combines metadata loading and PDF downloading into a single pipeline:
1. Loads arXiv metadata from JSONL file into PostgreSQL
2. Runs database migrations (including pdf_content BYTEA column)
3. Downloads PDFs from arXiv.org and stores them directly in the database
4. Uses concurrent workers for efficient PDF downloading
5. Respects rate limits and provides progress tracking

Usage:
    python scripts/process_arxiv.py --limit 5000 --workers 4
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from sqlalchemy.dialects.postgresql import insert

# Add project root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(override=True)

from database.database import DatabaseManager, get_database_url_from_env
from database.models import ArxivPaper, ArxivCategory, ArxivPaperCategory


class ArxivProcessor:
    def __init__(
        self,
        db_url: Optional[str] = None,
        arxiv_json_path: Optional[str] = None,
        download_delay: float = 3.0,
        workers: int = 4,
        retries: int = 3,
        timeout: float = 30.0,
    ):
        """Initialize the ArXiv processor."""
        self.db_url = db_url or get_database_url_from_env()
        self.arxiv_json_path = arxiv_json_path or os.getenv('ARXIV_DATA_PATH', './arxiv-metadata-oai-snapshot.json')
        self.download_delay = download_delay
        self.workers = workers
        self.retries = retries
        self.timeout = timeout

        self.db_manager = DatabaseManager(db_url=self.db_url)
        self.db_manager.create_engine_and_session()

        print(f"✓ Database: {self.db_manager.engine.url.render_as_string(hide_password=True)}")
        print(f"✓ ArXiv metadata: {self.arxiv_json_path}")
        print(f"✓ Workers: {self.workers}, Delay: {self.download_delay}s")

    def run_migrations(self):
        """Run Alembic migrations to ensure schema is up to date."""
        print("\n" + "="*60)
        print("Running database migrations...")
        print("="*60)

        import subprocess
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        if result.returncode != 0:
            print(f"Migration error: {result.stderr}")
            raise Exception("Failed to run migrations")

        print("✓ Migrations completed")

    def parse_record(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single JSONL record."""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def load_metadata(self, limit: Optional[int] = None, batch_size: int = 2000):
        """
        Load metadata from JSONL file into PostgreSQL.

        Returns:
            List of paper IDs that were loaded
        """
        print("\n" + "="*60)
        print("Loading metadata into database...")
        print("="*60)

        if not Path(self.arxiv_json_path).exists():
            raise FileNotFoundError(f"ArXiv metadata file not found: {self.arxiv_json_path}")

        loaded_ids = []
        total_processed = 0

        with open(self.arxiv_json_path, 'r', encoding='utf-8') as f:
            batch_papers = []
            batch_categories = set()

            for line in tqdm(f, desc="Loading metadata", unit=" records"):
                if limit and total_processed >= limit:
                    break

                record = self.parse_record(line)
                if not record:
                    continue

                paper_id = record.get('id')
                if not paper_id:
                    continue

                # Parse categories
                categories_str = record.get('categories', '')
                categories_list = [cat.strip() for cat in categories_str.split() if cat.strip()]

                # Prepare paper data
                paper_data = {
                    'id': paper_id,
                    'submitter': record.get('submitter'),
                    'authors': record.get('authors'),
                    'authors_parsed': record.get('authors_parsed'),
                    'title': record.get('title', ''),
                    'abstract': record.get('abstract'),
                    'comments': record.get('comments'),
                    'journal_ref': record.get('journal-ref'),
                    'doi': record.get('doi'),
                    'report_no': record.get('report-no'),
                    'categories': categories_str,
                    'license': record.get('license'),
                    'versions': record.get('versions'),
                    'update_date': datetime.strptime(record['update_date'], '%Y-%m-%d').date() if record.get('update_date') else None,
                }

                batch_papers.append(paper_data)
                batch_categories.update(categories_list)
                loaded_ids.append(paper_id)
                total_processed += 1

                # Process batch
                if len(batch_papers) >= batch_size:
                    self._insert_batch(batch_papers, batch_categories)
                    batch_papers = []
                    batch_categories = set()

            # Insert remaining records
            if batch_papers:
                self._insert_batch(batch_papers, batch_categories)

        print(f"✓ Loaded {total_processed} metadata records")
        return loaded_ids

    def _insert_batch(self, papers: List[Dict], categories: set):
        """Insert a batch of papers and categories."""
        with self.db_manager.get_session() as session:
            # Insert categories
            if categories:
                for cat in categories:
                    stmt = insert(ArxivCategory).values(category=cat).on_conflict_do_nothing(index_elements=['category'])
                    session.execute(stmt)

            # Upsert papers
            if papers:
                stmt = insert(ArxivPaper).values(papers)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['id'],
                    set_={
                        'submitter': stmt.excluded.submitter,
                        'authors': stmt.excluded.authors,
                        'authors_parsed': stmt.excluded.authors_parsed,
                        'title': stmt.excluded.title,
                        'abstract': stmt.excluded.abstract,
                        'comments': stmt.excluded.comments,
                        'journal_ref': stmt.excluded.journal_ref,
                        'doi': stmt.excluded.doi,
                        'report_no': stmt.excluded.report_no,
                        'categories': stmt.excluded.categories,
                        'license': stmt.excluded.license,
                        'versions': stmt.excluded.versions,
                        'update_date': stmt.excluded.update_date,
                    }
                )
                session.execute(stmt)

                # Handle paper-category relationships
                session.query(ArxivPaperCategory).filter(
                    ArxivPaperCategory.paper_id.in_([p['id'] for p in papers])
                ).delete(synchronize_session=False)

                paper_cat_links = []
                for paper in papers:
                    cats = [c.strip() for c in paper['categories'].split() if c.strip()]
                    for cat in cats:
                        paper_cat_links.append({'paper_id': paper['id'], 'category': cat})

                if paper_cat_links:
                    session.execute(insert(ArxivPaperCategory).values(paper_cat_links))

            session.commit()

    def download_pdf_to_db(self, paper_id: str, session_http: requests.Session) -> tuple[str, bool, Optional[str], Optional[bytes]]:
        """
        Download a PDF and return the result.

        Returns:
            (paper_id, success, error_message, pdf_bytes)
        """
        url = urljoin("https://arxiv.org/", f"pdf/{paper_id}.pdf")
        last_error = None

        for attempt in range(1, self.retries + 1):
            try:
                resp = session_http.get(url, timeout=self.timeout, stream=True)
                if resp.status_code == 200:
                    pdf_bytes = resp.content
                    return (paper_id, True, None, pdf_bytes)
                else:
                    last_error = f"HTTP {resp.status_code}"
            except Exception as e:
                last_error = str(e)

            if attempt < self.retries:
                time.sleep(1)

        return (paper_id, False, last_error, None)

    def download_pdfs_concurrent(self, paper_ids: List[str]):
        """
        Download PDFs concurrently and store them in the database.
        """
        print("\n" + "="*60)
        print(f"Downloading {len(paper_ids)} PDFs with {self.workers} workers...")
        print("="*60)

        success_count = 0
        error_count = 0
        total_bytes = 0

        def worker(paper_id: str):
            """Worker function for downloading a single PDF."""
            session_http = requests.Session()
            session_http.headers.update({"User-Agent": "arxiv-processor/1.0"})

            # Rate limiting: sleep before download
            time.sleep(self.download_delay / self.workers)

            result = self.download_pdf_to_db(paper_id, session_http)
            return result

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(worker, pid): pid for pid in paper_ids}

            with tqdm(total=len(paper_ids), desc="Downloading PDFs", unit=" pdfs") as pbar:
                for future in as_completed(futures):
                    paper_id, success, error, pdf_bytes = future.result()

                    # Update database
                    with self.db_manager.get_session() as session:
                        paper = session.query(ArxivPaper).filter_by(id=paper_id).first()
                        if paper:
                            paper.pdf_download_attempted_at = datetime.utcnow()

                            if success:
                                paper.pdf_content = pdf_bytes
                                paper.pdf_downloaded = True
                                paper.pdf_download_error = None
                                success_count += 1
                                total_bytes += len(pdf_bytes)
                            else:
                                paper.pdf_download_error = error
                                error_count += 1

                            session.commit()

                    pbar.update(1)

        print(f"\n{'='*60}")
        print(f"Download Summary:")
        print(f"  Success: {success_count}")
        print(f"  Errors: {error_count}")
        print(f"  Total data: {total_bytes / (1024*1024):.2f} MB")
        print(f"{'='*60}")

        # Sync status: verify pdf_content and pdf_downloaded are consistent
        self._sync_pdf_status()

    def _sync_pdf_status(self):
        """
        Sync pdf_downloaded flag based on pdf_content presence.
        Ensures consistency between pdf_content and pdf_downloaded flag.
        """
        print("\nSyncing PDF status...")

        with self.db_manager.get_session() as session:
            # Find papers where pdf_content exists but pdf_downloaded is False
            need_true = session.query(ArxivPaper).filter(
                ArxivPaper.pdf_content.isnot(None),
                ArxivPaper.pdf_downloaded.is_(False)
            ).count()

            if need_true > 0:
                session.query(ArxivPaper).filter(
                    ArxivPaper.pdf_content.isnot(None),
                    ArxivPaper.pdf_downloaded.is_(False)
                ).update({'pdf_downloaded': True}, synchronize_session=False)

            # Find papers where pdf_content is None but pdf_downloaded is True
            need_false = session.query(ArxivPaper).filter(
                ArxivPaper.pdf_content.is_(None),
                ArxivPaper.pdf_downloaded.is_(True)
            ).count()

            if need_false > 0:
                session.query(ArxivPaper).filter(
                    ArxivPaper.pdf_content.is_(None),
                    ArxivPaper.pdf_downloaded.is_(True)
                ).update({'pdf_downloaded': False}, synchronize_session=False)

            session.commit()

            if need_true > 0 or need_false > 0:
                print(f"✓ Synced {need_true + need_false} records (set true: {need_true}, set false: {need_false})")
            else:
                print("✓ All records are already in sync")

    def process(self, limit: Optional[int] = None, skip_metadata: bool = False, skip_download: bool = False):
        """
        Main processing pipeline.

        Args:
            limit: Maximum number of papers to process
            skip_metadata: Skip metadata loading step
            skip_download: Skip PDF download step
        """
        # Step 1: Run migrations
        if not skip_metadata:
            self.run_migrations()

        # Step 2: Load metadata
        if skip_metadata:
            print("\nSkipping metadata loading...")
            with self.db_manager.get_session() as session:
                query = session.query(ArxivPaper.id).filter(
                    ArxivPaper.pdf_downloaded.is_(False)
                )
                if limit:
                    query = query.limit(limit)
                paper_ids = [row[0] for row in query.all()]
        else:
            paper_ids = self.load_metadata(limit=limit)

        if not paper_ids:
            print("No papers to process.")
            return

        # Step 3: Download PDFs
        if not skip_download:
            self.download_pdfs_concurrent(paper_ids)
        else:
            print("\nSkipping PDF downloads...")

        print("\n✓ Processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Unified ArXiv metadata and PDF processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 5000 papers (metadata + PDFs)
  python scripts/process_arxiv.py --limit 5000

  # Process with 8 concurrent workers
  python scripts/process_arxiv.py --limit 1000 --workers 8

  # Only download PDFs for existing metadata
  python scripts/process_arxiv.py --skip-metadata --limit 500

  # Only load metadata, skip downloads
  python scripts/process_arxiv.py --skip-download --limit 10000
        """
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of papers to process (default: all)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of concurrent download workers (default: 4)'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=None,
        help='Delay between downloads in seconds (default: from .env DOWNLOAD_DELAY or 3.0)'
    )

    parser.add_argument(
        '--arxiv-json',
        default=None,
        help='Path to arXiv metadata JSONL file (default: from .env ARXIV_DATA_PATH)'
    )

    parser.add_argument(
        '--skip-metadata',
        action='store_true',
        help='Skip metadata loading, only download PDFs'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip PDF downloads, only load metadata'
    )

    parser.add_argument(
        '--db-url',
        default=None,
        help='Database URL (default: from .env SQLALCHEMY_URL)'
    )

    args = parser.parse_args()

    # Get download delay from args or env
    delay = args.delay or float(os.getenv('DOWNLOAD_DELAY', '3.0'))

    # Create processor
    processor = ArxivProcessor(
        db_url=args.db_url,
        arxiv_json_path=args.arxiv_json,
        download_delay=delay,
        workers=args.workers
    )

    # Run processing pipeline
    processor.process(
        limit=args.limit,
        skip_metadata=args.skip_metadata,
        skip_download=args.skip_download
    )


if __name__ == '__main__':
    main()
