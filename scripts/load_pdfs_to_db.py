#!/usr/bin/env python3
"""
Load PDF files from filesystem into PostgreSQL database

This script reads PDF files from the filesystem and stores them in the
pdf_content column (BYTEA) of the arxiv_papers table.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(override=True)

from sqlalchemy import update, select
from database.database import DatabaseManager, get_database_url_from_env
from database.models import ArxivPaper


class PDFLoader:
    def __init__(
        self,
        db_url: Optional[str] = None,
        db_host: Optional[str] = None,
        db_port: Optional[int] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
    ):
        """Initialize the PDF loader with database connection parameters."""
        # If no explicit URL provided, use environment variables
        if db_url is None:
            db_url = get_database_url_from_env()

        # Read from env for parameters not explicitly provided
        if db_host is None:
            db_host = os.getenv('DB_HOST', '')
        if db_port is None:
            db_port = int(os.getenv('DB_PORT', '5432'))
        if db_name is None:
            db_name = os.getenv('DB_NAME', 'arxiv')
        if db_user is None:
            db_user = os.getenv('DB_USER', 'postgres')
        if db_password is None:
            db_password = os.getenv('DB_PASSWORD', '')

        self.db_manager = DatabaseManager(
            db_url=db_url,
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password
        )

    def load_pdfs_from_filesystem(
        self,
        pdf_directory: str,
        batch_size: int = 100,
        limit: Optional[int] = None,
        skip_existing: bool = True
    ):
        """
        Load PDF files from filesystem into database.

        Args:
            pdf_directory: Directory containing PDF files
            batch_size: Number of PDFs to load in each transaction
            limit: Maximum number of PDFs to load (None for all)
            skip_existing: Skip files that already have pdf_content in DB
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            print(f"Error: Directory {pdf_directory} does not exist")
            return

        # Get list of PDF files
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        total_files = len(pdf_files)

        if limit:
            pdf_files = pdf_files[:limit]

        print(f"Found {total_files} PDF files")
        print(f"Processing {len(pdf_files)} files (limit={limit or 'all'})")

        with self.db_manager.get_session() as session:
            loaded_count = 0
            skipped_count = 0
            error_count = 0
            total_bytes = 0

            for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
                try:
                    # Extract paper ID from filename (e.g., 0704.0001.pdf -> 0704.0001)
                    paper_id = pdf_file.stem

                    # Check if paper exists in database
                    paper = session.query(ArxivPaper).filter_by(id=paper_id).first()

                    if not paper:
                        print(f"\nWarning: Paper {paper_id} not found in database, skipping")
                        skipped_count += 1
                        continue

                    # Skip if already has content and skip_existing is True
                    if skip_existing and paper.pdf_content is not None:
                        skipped_count += 1
                        continue

                    # Read PDF file
                    with open(pdf_file, 'rb') as f:
                        pdf_bytes = f.read()

                    file_size = len(pdf_bytes)
                    total_bytes += file_size

                    # Update database
                    paper.pdf_content = pdf_bytes
                    paper.pdf_downloaded = True

                    loaded_count += 1

                    # Commit in batches
                    if loaded_count % batch_size == 0:
                        session.commit()
                        tqdm.write(f"Committed batch: {loaded_count} loaded, {skipped_count} skipped")

                except Exception as e:
                    print(f"\nError processing {pdf_file}: {e}")
                    error_count += 1
                    session.rollback()
                    continue

            # Final commit
            if loaded_count % batch_size != 0:
                session.commit()

        # Summary
        print(f"\n{'='*60}")
        print(f"PDF Loading Summary:")
        print(f"  Total files found: {total_files}")
        print(f"  Successfully loaded: {loaded_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Errors: {error_count}")
        print(f"  Total data loaded: {total_bytes / (1024*1024):.2f} MB")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Load PDF files from filesystem into PostgreSQL database'
    )

    parser.add_argument(
        '--pdf-directory',
        default=None,
        help='Directory containing PDF files (default: reads from .env PDF_BASE_PATH)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of PDFs to load per transaction (default: 100)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of PDFs to load (default: all)'
    )

    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Reload PDFs even if they already exist in database'
    )

    # Database connection parameters (optional, defaults to .env values)
    parser.add_argument('--db-url', default=None, help='Full database URL (overrides .env and other DB params)')
    parser.add_argument('--db-host', default=None, help='Database host (overrides .env; empty for Unix socket)')
    parser.add_argument('--db-port', type=int, default=None, help='Database port (overrides .env)')
    parser.add_argument('--db-name', default=None, help='Database name (overrides .env)')
    parser.add_argument('--db-user', default=None, help='Database user (overrides .env)')
    parser.add_argument('--db-password', default=None, help='Database password (overrides .env)')

    args = parser.parse_args()

    # Get PDF directory from args or .env
    pdf_directory = args.pdf_directory or os.getenv('PDF_BASE_PATH', './arxiv_pdfs')

    # Create loader (will use .env values if args are None)
    loader = PDFLoader(
        db_url=args.db_url,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )

    # Load PDFs
    loader.load_pdfs_from_filesystem(
        pdf_directory=pdf_directory,
        batch_size=args.batch_size,
        limit=args.limit,
        skip_existing=not args.no_skip_existing
    )


if __name__ == '__main__':
    main()
