#!/usr/bin/env python3
"""
OCR client that fetches PDFs from database and stores results back to database.

- Pulls papers with pdf_content (PDF stored in DB) that haven't been OCR processed yet
- Pre-renders PDF pages to images and sends them to the FastAPI OCR service
- Pipelines CPU rendering of the next PDF while GPU processes the current one (queue size 1)
- Caps rendering to 100 pages by default to avoid OOM on very long PDFs
- Saves OCR results directly to the database (ocr_results JSONB column)
- Tracks processing status via ocr_processed, ocr_processed_at, ocr_error columns
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.database import DatabaseManager  # noqa: E402
from database.models import ArxivPaper  # noqa: E402

load_dotenv(override=True)

DEFAULT_RENDER_DPI = 200
RENDER_PAGE_LIMIT = 100
DEFAULT_PAGE_BATCH_SIZE = 8


def get_service_info(service_url: str) -> dict:
    """Query OCR service for configuration info including worker count."""
    try:
        # Convert endpoint URL to /info URL
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


def _get_pdf_total_pages(pdf_bytes: bytes) -> int:
    """Return total page count for a PDF."""
    reader = PdfReader(BytesIO(pdf_bytes))
    return len(reader.pages)


def _render_pdf_to_images(
    pdf_bytes: bytes,
    target_pages: int,
    dpi: int = DEFAULT_RENDER_DPI
) -> Dict[int, bytes]:
    """Render PDF pages to PNG bytes once client-side to avoid repeated server renders."""
    images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=target_pages)
    rendered = {}

    for idx, image in enumerate(images):
        with BytesIO() as buf:
            image.save(buf, format="PNG")
            rendered[idx] = buf.getvalue()
        image.close()

    return rendered


def _render_pdf_chunk(
    pdf_bytes: bytes,
    start_page: int,
    end_page: int,
    dpi: int = DEFAULT_RENDER_DPI
) -> Dict[int, bytes]:
    """Render a contiguous chunk of PDF pages [start_page, end_page] (0-based inclusive) to PNG bytes."""
    images = convert_from_bytes(
        pdf_bytes,
        dpi=dpi,
        first_page=start_page + 1,
        last_page=end_page + 1,
    )
    rendered = {}
    for idx, image in enumerate(images, start=start_page):
        with BytesIO() as buf:
            image.save(buf, format="PNG")
            rendered[idx] = buf.getvalue()
        image.close()
    return rendered


def _prepare_render_job(
    paper_id: str,
    pdf_bytes: bytes,
    max_pages: Optional[int],
    dpi: int,
) -> dict:
    """Prepare a PDF for GPU processing by gathering metadata (page counts). Runs in a thread."""
    try:
        total_pages = _get_pdf_total_pages(pdf_bytes)
    except Exception as e:
        return {"ok": False, "error": f"could not determine total pages: {e}"}

    # The "goal" target is everything, unless the user explicitly wants to partial-process via --pages
    target_pages = min(total_pages, max_pages) if max_pages else total_pages

    if target_pages <= 0:
        return {"ok": False, "error": "no pages to process after applying limits"}

    return {
        "ok": True,
        "paper_id": paper_id,
        "total_pages": total_pages,
        "target_pages": target_pages,
        "render_dpi": dpi,
    }


def _process_single_image(
    paper_id: str,
    page_num: int,
    image_bytes: bytes,
    service_url: str,
    dpi: int,
) -> Tuple[int, Optional[dict], Optional[str]]:
    """Send a pre-rendered page image to the OCR service. Returns (page_num, result, error)."""
    if not image_bytes:
        return (page_num, None, "missing rendered image bytes")

    try:
        files = {"file": (f"{paper_id}_page_{page_num + 1}.png", image_bytes, "image/png")}
        data = {"page": page_num, "dpi": dpi}
        resp = requests.post(service_url, files=files, data=data, timeout=6000)
    except Exception as e:
        return (page_num, None, f"request failed: {e}")

    if resp.status_code != 200:
        return (page_num, None, f"service error {resp.status_code}: {resp.text}")

    try:
        payload = resp.json()
    except Exception as e:
        return (page_num, None, f"failed to parse JSON: {e}")

    page_result = payload.get("result")

    return (page_num, page_result, None)


def process_pdf_to_db(
    db: DatabaseManager,
    paper_id: str,
    pdf_bytes: bytes,
    service_url: str,
    max_pages: int = None,
    gpu_workers: int = 1,
    total_pages: Optional[int] = None,
    target_pages: Optional[int] = None,
    render_dpi: int = DEFAULT_RENDER_DPI,
    render_page_limit: int = RENDER_PAGE_LIMIT,
    page_batch_size: int = DEFAULT_PAGE_BATCH_SIZE,
) -> bool:
    """Process a single PDF with concurrent page processing."""

    # Load existing partial results for resume support
    existing_data = get_partial_results(db, paper_id)
    results_dict = {}
    completed_pages = set()
    total_pages = total_pages or existing_data.get("total_pages")

    # Restore previously processed pages
    if existing_data.get("results"):
        for i, r in enumerate(existing_data["results"]):
            page_num = _extract_page_index(r, i)
            results_dict[page_num] = r
            completed_pages.add(page_num)
        if completed_pages:
            print(f"  Resuming: {len(completed_pages)} pages already completed")
            save_ocr_results(db, paper_id, existing_data, clear_error=True)

    # Get total pages locally to avoid server-side PDF rendering
    if total_pages is None:
        try:
            total_pages = _get_pdf_total_pages(pdf_bytes)
            print(f"  Total pages: {total_pages}")
        except Exception as e:
            error_msg = f"could not determine total pages: {e}"
            print(f"  Error: {error_msg}")
            save_ocr_results(db, paper_id, {}, error=error_msg)
            return False

    if not total_pages:
        error_msg = "could not determine total pages"
        print(f"  Error: {error_msg}")
        save_ocr_results(db, paper_id, {}, error=error_msg)
        return False

    # Determine "goal" pages (what we ultimately want to finish)
    effective_target = target_pages
    if effective_target is None:
        effective_target = min(total_pages, max_pages) if max_pages else total_pages

    if effective_target <= 0:
        error_msg = "no pages to process after applying limits"
        print(f"  Error: {error_msg}")
        save_ocr_results(db, paper_id, {}, error=error_msg)
        return False

    # Determine which pages need processing right now
    # We filter out already completed pages
    all_pages_needed = [p for p in range(effective_target) if p not in completed_pages]

    if not all_pages_needed:
        print(f"  All pages already completed")
        save_ocr_results(db, paper_id, _build_results(results_dict, total_pages), completed=True)
        return True

    # APPLY BATCH LIMIT: Only take the first N needed pages for this run
    if render_page_limit and len(all_pages_needed) > render_page_limit:
        print(f"  Batching: processing next {render_page_limit} pages (of {len(all_pages_needed)} remaining)")
        pages_to_process = all_pages_needed[:render_page_limit]
        is_partial_run = True
    else:
        pages_to_process = all_pages_needed
        is_partial_run = False

    print(
        f"  Processing {len(pages_to_process)} pages with up to "
        f"{min(gpu_workers, page_batch_size)} concurrent worker(s) per batch..."
    )

    processed = 0
    errors = []

    def iter_contiguous_runs(pages):
        if not pages:
            return
        run = [pages[0]]
        for p in pages[1:]:
            if p == run[-1] + 1:
                run.append(p)
            else:
                yield run
                run = [p]
        if run:
            yield run

    # Process in contiguous batches to avoid re-rendering already completed pages
    for run in iter_contiguous_runs(sorted(pages_to_process)):
        for i in range(0, len(run), page_batch_size):
            chunk = run[i : i + page_batch_size]
            start_page, end_page = chunk[0], chunk[-1]

            try:
                page_images = _render_pdf_chunk(pdf_bytes, start_page, end_page, dpi=render_dpi)
            except Exception as e:
                error_msg = f"failed to render pages {start_page}-{end_page}: {e}"
                print(f"  Error: {error_msg}")
                errors.append(error_msg)
                continue

            # Send chunk with limited concurrency
            concurrency = min(gpu_workers, page_batch_size, len(chunk))
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        _process_single_image,
                        paper_id,
                        page_num,
                        page_images.get(page_num),
                        service_url,
                        render_dpi,
                    ): page_num
                    for page_num in chunk
                }

                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        pg, result, error = future.result()
                        if error:
                            errors.append(f"page {pg}: {error}")
                            print(f"  Page {pg + 1}/{total_pages}: ERROR - {error}")
                        elif result is not None:
                            results_dict[pg] = result
                            completed_pages.add(pg)
                            processed += 1
                            print(
                                f"  Page {pg + 1}/{total_pages}: OK "
                                f"({processed}/{len(pages_to_process)})"
                            )

                            # Save incrementally (without completed flag) for resume support
                            save_ocr_results(db, paper_id, _build_results(results_dict, total_pages))
                    except Exception as e:
                        errors.append(f"page {page_num}: exception {e}")
                        print(f"  Page {page_num + 1}/{total_pages}: EXCEPTION - {e}")

    # Final save logic
    # We only mark completed=True if we have actually finished ALL pages for the target,
    # AND there are no errors in this run (or previously if we tracked them, but mainly this run).
    # If is_partial_run is True, we definitely aren't done.
    
    total_finished_now = len(completed_pages)
    actually_all_done = total_finished_now >= effective_target

    if errors:
        error_msg = "; ".join(errors[:3])
        if len(errors) > 3:
            error_msg += f" (+{len(errors) - 3} more)"
        save_ocr_results(db, paper_id, _build_results(results_dict, total_pages), error=error_msg)
        print(f"  Run completed with {len(errors)} error(s)")
        return False

    elif actually_all_done:
        save_ocr_results(db, paper_id, _build_results(results_dict, total_pages), completed=True)
        print(f"  Completed successfully (all {total_finished_now} pages)")
        return True
        
    else:
        # We finished this batch, but the PDF isn't fully done yet
        save_ocr_results(db, paper_id, _build_results(results_dict, total_pages))
        if is_partial_run:
             print(f"  Batch complete: {processed} pages processed. Reschedule to finish remaining pages.")
        else:
             print(f"  Partial: {total_finished_now}/{effective_target} pages completed")
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
    parser.add_argument(
        "--service-url",
        default="http://localhost:8000/ocr_image",
        help="OCR service endpoint (use /ocr_image to send pre-rendered pages)",
    )
    parser.add_argument("--pages", type=int, default=None, help="Max pages per PDF (default: all)")
    parser.add_argument("--limit", type=int, default=10, help="Max number of PDFs to process")
    parser.add_argument("--offset", type=int, default=0, help="Offset for DB query")
    parser.add_argument("--gpu-workers", type=int, default=None,
                        help="Number of concurrent pages to process (default: auto-detect from OCR service)")
    parser.add_argument("--retry-errors", action="store_true", help="Retry papers that previously failed with errors")
    parser.add_argument("--render-page-limit", type=int, default=RENDER_PAGE_LIMIT,
                        help="Max pages to render per PDF to avoid OOM (default: 100)")
    parser.add_argument("--page-batch-size", type=int, default=DEFAULT_PAGE_BATCH_SIZE,
                        help="Max pages to render/process per batch to reduce GPU/CPU spikes (default: 8)")
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

    # Pipeline: precompute next PDF metadata while GPU processes current PDF pages
    render_executor = ThreadPoolExecutor(max_workers=2)

    def schedule_render(p_id: str, pdf_bytes: Optional[bytes]):
        if not pdf_bytes:
            return None
        return render_executor.submit(
            _prepare_render_job,
            p_id,
            pdf_bytes,
            args.pages,
            DEFAULT_RENDER_DPI,
        )

    current_pdf_bytes = None
    current_render_future = None

    if paper_ids:
        first_id = paper_ids[0]
        current_pdf_bytes = get_pdf_content(db, first_id)
        if current_pdf_bytes:
            current_render_future = schedule_render(first_id, current_pdf_bytes)
    for i, paper_id in enumerate(paper_ids, 1):
        print(f"\n[{i}/{len(paper_ids)}] {paper_id}")
        print("-" * 50)

        # Ensure we have a render job for this paper; try to start one if missing
        if current_render_future is None:
            current_pdf_bytes = current_pdf_bytes or get_pdf_content(db, paper_id)
            if current_pdf_bytes:
                current_render_future = schedule_render(paper_id, current_pdf_bytes)

        # If still no render job, skip but prefetch the next paper to keep pipeline moving
        if current_render_future is None:
            next_pdf_bytes = None
            next_render_future = None
            if i < len(paper_ids):
                next_id = paper_ids[i]
                next_pdf_bytes = get_pdf_content(db, next_id)
                if next_pdf_bytes:
                    next_render_future = schedule_render(next_id, next_pdf_bytes)
                else:
                    print(f"  (prefetch) No PDF content in database for {next_id}")

            print("  No PDF content in database, skipping")
            fail_count += 1
            current_pdf_bytes = next_pdf_bytes
            current_render_future = next_render_future
            continue

        # Wait for render to finish
        render_data = current_render_future.result()

        # Prefetch next PDF render while GPU works on this one
        next_pdf_bytes = None
        next_render_future = None
        if i < len(paper_ids):
            next_id = paper_ids[i]
            next_pdf_bytes = get_pdf_content(db, next_id)
            if next_pdf_bytes:
                next_render_future = schedule_render(next_id, next_pdf_bytes)
            else:
                print(f"  (prefetch) No PDF content in database for {next_id}")

        if not render_data.get("ok"):
            error_msg = render_data.get("error", "rendering failed")
            print(f"  Render error: {error_msg}")
            save_ocr_results(db, paper_id, {}, error=error_msg)
            fail_count += 1
            current_pdf_bytes = next_pdf_bytes
            current_render_future = next_render_future
            continue

        print(
            f"  Prepared metadata for {render_data['target_pages']}/{render_data['total_pages']} page(s) at {render_data['render_dpi']} DPI"
        )
        if render_data.get("capped"):
            print(f"  Note: capped pages due to render-page-limit={args.render_page_limit}")

        success = process_pdf_to_db(
            db=db,
            paper_id=paper_id,
            pdf_bytes=current_pdf_bytes or b"",
            service_url=args.service_url,
            max_pages=args.pages,
            gpu_workers=gpu_workers,
            total_pages=render_data["total_pages"],
            target_pages=render_data["target_pages"],
            render_dpi=render_data["render_dpi"],
            render_page_limit=args.render_page_limit,
            page_batch_size=args.page_batch_size,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

        # Move pipeline forward
        current_pdf_bytes = next_pdf_bytes
        current_render_future = next_render_future

    render_executor.shutdown(wait=True)

    print(f"\n{'='*50}")
    print(f"Done: {success_count} succeeded, {fail_count} failed")
    print('='*50)


if __name__ == "__main__":
    main()
