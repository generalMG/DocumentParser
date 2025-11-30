#!/usr/bin/env python3
"""
ArXiv Metadata Database Loader using SQLAlchemy

This script loads arXiv metadata from a JSONL file into PostgreSQL.
It processes 1.7M+ records efficiently using batch inserts with SQLAlchemy ORM.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import argparse
from dotenv import load_dotenv

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(override=True)

from sqlalchemy.dialects.postgresql import insert

from database.database import DatabaseManager
from database.models import ArxivPaper, ArxivCategory, ArxivPaperCategory


class ArxivLoader:
    def __init__(
        self,
        db_url: Optional[str] = None,
        db_host: str = "",
        db_port: int = 5432,
        db_name: str = "arxiv",
        db_user: str = "postgres",
        db_password: str = "",
        pdf_base_path: str = None
    ):
        """Initialize the ArXiv loader with database connection parameters."""
        self.pdf_base_path = pdf_base_path or os.getenv('PDF_BASE_PATH', './arxiv_pdfs')
        self.db_manager = DatabaseManager(
            db_url=db_url,
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password
        )
        self.db_manager.create_engine_and_session()
        engine_url = self.db_manager.engine.url.render_as_string(hide_password=True)
        print(f"✓ Connected to database: {engine_url}")

    def close(self):
        """Close database connection."""
        self.db_manager.close()

    def generate_pdf_path(self, arxiv_id: str) -> str:
        """
        Generate the PDF file path for a given arXiv ID.

        Args:
            arxiv_id: ArXiv paper ID (e.g., "0704.0001")

        Returns:
            Full path to the PDF file
        """
        return f"{self.pdf_base_path}/{arxiv_id}.pdf"

    def parse_record(self, json_line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single JSON line and prepare it for database insertion.

        Args:
            json_line: Raw JSON string from the JSONL file

        Returns:
            Dictionary with processed data or None if parsing fails
        """
        try:
            data = json.loads(json_line.strip())

            # Generate PDF path
            pdf_path = self.generate_pdf_path(data['id'])

            # Parse categories (space-separated)
            categories_raw = data.get('categories', '')
            category_list = categories_raw.split() if categories_raw else []

            return {
                'id': data.get('id'),
                'submitter': data.get('submitter'),
                'authors': data.get('authors'),
                'authors_parsed': data.get('authors_parsed', []),
                'title': data.get('title'),
                'abstract': data.get('abstract'),
                'comments': data.get('comments'),
                'journal_ref': data.get('journal-ref'),
                'doi': data.get('doi'),
                'report_no': data.get('report-no'),
                'categories': categories_raw,
                'license': data.get('license'),
                'versions': data.get('versions', []),
                'update_date': data.get('update_date'),
                'pdf_path': pdf_path,
                'category_list': category_list,
                # PDF download tracking (default to not downloaded)
                'pdf_downloaded': False,
                'pdf_download_attempted_at': None,
                'pdf_download_error': None
            }
        except Exception as e:
            print(f"✗ Error parsing record: {e}")
            return None

    def insert_papers_batch(self, session, batch: List[Dict[str, Any]]):
        """
        Insert a batch of papers using upsert.

        Args:
            session: SQLAlchemy session
            batch: List of parsed records
        """
        if not batch:
            return

        # Prepare data for bulk insert (exclude category_list)
        papers_data = []
        for record in batch:
            paper_dict = {k: v for k, v in record.items() if k != 'category_list'}
            papers_data.append(paper_dict)

        # Use PostgreSQL INSERT ... ON CONFLICT DO UPDATE
        stmt = insert(ArxivPaper).values(papers_data)
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
                'pdf_path': stmt.excluded.pdf_path,
                'updated_at': datetime.utcnow()
                # Note: pdf_downloaded, pdf_download_attempted_at, pdf_download_error
                # are NOT updated on conflict to preserve download tracking status
            }
        )
        session.execute(stmt)

    def insert_categories_batch(self, session, batch: List[Dict[str, Any]]):
        """
        Insert categories and paper-category relationships.

        Args:
            session: SQLAlchemy session
            batch: List of parsed records with category_list
        """
        # Collect all unique categories from the batch
        all_categories = set()
        for record in batch:
            all_categories.update(record.get('category_list', []))

        # Insert categories (ignore conflicts)
        if all_categories:
            category_data = [{'category': cat} for cat in all_categories]
            stmt = insert(ArxivCategory).values(category_data)
            stmt = stmt.on_conflict_do_nothing(index_elements=['category'])
            session.execute(stmt)

        # Delete existing paper-category relationships for these papers
        paper_ids = [record['id'] for record in batch]
        session.query(ArxivPaperCategory).filter(
            ArxivPaperCategory.paper_id.in_(paper_ids)
        ).delete(synchronize_session=False)

        # Insert paper-category relationships
        paper_category_data = []
        for record in batch:
            paper_id = record['id']
            for category in record.get('category_list', []):
                paper_category_data.append({
                    'paper_id': paper_id,
                    'category': category
                })

        if paper_category_data:
            stmt = insert(ArxivPaperCategory).values(paper_category_data)
            stmt = stmt.on_conflict_do_nothing()
            session.execute(stmt)

    def load_data(
        self,
        json_file: str,
        batch_size: int = 1000,
        max_records: Optional[int] = None
    ):
        """
        Load data from JSONL file into PostgreSQL database.

        Args:
            json_file: Path to the JSONL file
            batch_size: Number of records to insert in each batch
            max_records: Maximum number of records to process (None for all)
        """
        batch = []
        total_processed = 0
        total_inserted = 0
        total_errors = 0

        print(f"\nLoading data from: {json_file}")
        print(f"Batch size: {batch_size}")
        print(f"PDF base path: {self.pdf_base_path}")
        print("-" * 60)

        start_time = datetime.now()

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_records and line_num > max_records:
                        break

                    record = self.parse_record(line)
                    if record:
                        batch.append(record)
                        total_processed += 1
                    else:
                        total_errors += 1

                    # Insert batch when it reaches batch_size
                    if len(batch) >= batch_size:
                        try:
                            with self.db_manager.session_scope() as session:
                                self.insert_papers_batch(session, batch)
                                self.insert_categories_batch(session, batch)

                            total_inserted += len(batch)

                            # Progress update
                            elapsed = (datetime.now() - start_time).total_seconds()
                            rate = total_processed / elapsed if elapsed > 0 else 0
                            print(f"Processed: {total_processed:,} | "
                                  f"Inserted: {total_inserted:,} | "
                                  f"Errors: {total_errors} | "
                                  f"Rate: {rate:.1f} rec/sec", end='\r')

                            batch = []
                        except Exception as e:
                            print(f"\n✗ Batch insert failed: {e}")
                            total_errors += len(batch)
                            batch = []

                # Insert remaining records
                if batch:
                    try:
                        with self.db_manager.session_scope() as session:
                            self.insert_papers_batch(session, batch)
                            self.insert_categories_batch(session, batch)

                        total_inserted += len(batch)
                    except Exception as e:
                        print(f"\n✗ Final batch insert failed: {e}")
                        total_errors += len(batch)

        except FileNotFoundError:
            print(f"✗ File not found: {json_file}")
            return
        except Exception as e:
            print(f"\n✗ Error during data loading: {e}")
            return

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "-" * 60)
        print(f"✓ Loading completed!")
        print(f"  Total processed: {total_processed:,}")
        print(f"  Total inserted: {total_inserted:,}")
        print(f"  Total errors: {total_errors}")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Average rate: {total_processed / duration:.1f} records/sec")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Load arXiv metadata from JSONL into PostgreSQL using SQLAlchemy"
    )
    parser.add_argument(
        '--db-url',
        default=os.getenv('SQLALCHEMY_URL'),
        help='Full SQLAlchemy DB URL (takes priority over other DB args)'
    )
    parser.add_argument(
        '--json-file',
        default=os.getenv('ARXIV_DATA_PATH', './arxiv-metadata-oai-snapshot.json'),
        help='Path to the JSONL file (default: from ARXIV_DATA_PATH env var)'
    )
    parser.add_argument(
        '--db-host',
        default=os.getenv('DB_HOST', ''),
        help='PostgreSQL host (empty for Unix socket)'
    )
    parser.add_argument(
        '--db-port',
        type=int,
        default=int(os.getenv('DB_PORT', '5432')),
        help='PostgreSQL port'
    )
    parser.add_argument(
        '--db-name',
        default=os.getenv('DB_NAME', 'arxiv'),
        help='Database name'
    )
    parser.add_argument(
        '--db-user',
        default=os.getenv('DB_USER', 'postgres'),
        help='Database user'
    )
    parser.add_argument(
        '--db-password',
        default=os.getenv('DB_PASSWORD', ''),
        help='Database password'
    )
    parser.add_argument(
        '--pdf-path',
        default=os.getenv('PDF_BASE_PATH', './arxiv_pdfs'),
        help='Base path for PDF files'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of records per batch insert'
    )
    parser.add_argument(
        '--max-records',
        type=int,
        default=None,
        help='Maximum number of records to process (for testing)'
    )

    args = parser.parse_args()

    # Create loader instance
    loader = ArxivLoader(
        db_url=args.db_url,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        pdf_base_path=args.pdf_path
    )

    try:
        # Load data
        loader.load_data(
            json_file=args.json_file,
            batch_size=args.batch_size,
            max_records=args.max_records
        )
    finally:
        # Close connection
        loader.close()


if __name__ == '__main__':
    main()
