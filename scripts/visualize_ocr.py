#!/usr/bin/env python3
"""
Visualize OCR results from database with bounding boxes and extracted text side by side.

- Pulls PDF and OCR results directly from database
- Shows bounding boxes on the PDF page
- Displays extracted text in a readable side panel
"""

import argparse
import io
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

try:
    from pdf2image import convert_from_bytes
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False
    print("Warning: pdf2image not installed. Install with: pip install pdf2image")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: PIL/Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.database import DatabaseManager
from database.models import ArxivPaper

load_dotenv(override=True)


# Color scheme for different label types
LABEL_COLORS = {
    'doc_title': (220, 53, 69),      # Red
    'header': (40, 167, 69),          # Green
    'footer': (0, 123, 255),          # Blue
    'text': (255, 152, 0),            # Orange
    'paragraph': (255, 152, 0),       # Orange (alias)
    'aside_text': (156, 39, 176),     # Purple
    'table': (0, 188, 212),           # Cyan
    'figure': (233, 30, 99),          # Pink
    'figure_caption': (121, 85, 72),  # Brown
    'table_caption': (0, 150, 136),   # Teal
    'equation': (255, 235, 59),       # Yellow
    'list': (139, 195, 74),           # Light Green
    'code': (96, 125, 139),           # Blue Grey
    'reference': (63, 81, 181),       # Indigo
}


def get_color_for_label(label: str) -> Tuple[int, int, int]:
    """Get a consistent color for each label type."""
    return LABEL_COLORS.get(label.lower(), (128, 128, 128))


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a good font, with fallbacks."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Windows/Fonts/arial.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def draw_boxes_on_image(
    image: Image.Image,
    boxes: List[dict],
    line_width: int = 3,
    font_size: int = 14,
) -> Image.Image:
    """Draw bounding boxes with labels on the image."""
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    font = get_font(font_size)

    for idx, box in enumerate(boxes):
        coords = box.get('coordinate', [])
        if len(coords) < 4:
            continue

        x1, y1, x2, y2 = coords[:4]
        label = box.get('label', 'unknown')
        score = float(box.get('score', 0))
        color = get_color_for_label(label)

        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)

        # Draw label with index
        label_text = f"{idx + 1}. {label} ({score:.2f})"
        bbox_text = draw.textbbox((0, 0), label_text, font=font)
        text_height = bbox_text[3] - bbox_text[1]
        text_width = bbox_text[2] - bbox_text[0]

        # Position label above the box
        label_y = max(0, y1 - text_height - 6)
        draw.rectangle(
            [(x1, label_y), (x1 + text_width + 8, label_y + text_height + 6)],
            fill=color
        )
        draw.text((x1 + 4, label_y + 3), label_text, fill=(255, 255, 255), font=font)

    return vis_image


def create_text_panel(
    boxes: List[dict],
    height: int,
    width: int = 600,
    font_size: int = 16,
    line_height: int = 24,
) -> Image.Image:
    """Create a text panel showing extracted content."""
    panel = Image.new('RGB', (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(panel)

    font = get_font(font_size)
    font_bold = get_font(font_size + 2)
    font_small = get_font(font_size - 2)

    # Title
    draw.text((20, 15), "Extracted Text", fill=(33, 33, 33), font=font_bold)
    draw.line([(20, 45), (width - 20, 45)], fill=(200, 200, 200), width=2)

    y_offset = 60
    max_y = height - 40
    wrap_width = (width - 50) // (font_size // 2)  # Approximate chars per line

    for idx, box in enumerate(boxes):
        if y_offset >= max_y:
            draw.text((20, y_offset), "... (content truncated)", fill=(128, 128, 128), font=font_small)
            break

        label = box.get('label', 'unknown')
        content = box.get('content', '').strip()
        color = get_color_for_label(label)

        # Draw index and label
        header = f"{idx + 1}. [{label}]"
        draw.text((20, y_offset), header, fill=color, font=font_bold)
        y_offset += line_height + 4

        if content:
            # Word wrap the content
            wrapped_lines = textwrap.wrap(content, width=wrap_width)
            for line in wrapped_lines[:8]:  # Limit lines per box
                if y_offset >= max_y:
                    break
                draw.text((30, y_offset), line, fill=(50, 50, 50), font=font)
                y_offset += line_height

            if len(wrapped_lines) > 8:
                draw.text((30, y_offset), "...", fill=(128, 128, 128), font=font_small)
                y_offset += line_height
        else:
            draw.text((30, y_offset), "(no text extracted)", fill=(150, 150, 150), font=font_small)
            y_offset += line_height

        y_offset += 15  # Space between entries

    return panel


def visualize_page(
    image: Image.Image,
    boxes: List[dict],
    text_panel_width: int = 600,
    font_size: int = 16,
) -> Image.Image:
    """Create side-by-side visualization: image with boxes + text panel."""
    # Draw boxes on image
    annotated = draw_boxes_on_image(image, boxes, line_width=3, font_size=14)

    # Create text panel
    text_panel = create_text_panel(
        boxes,
        height=annotated.height,
        width=text_panel_width,
        font_size=font_size,
    )

    # Combine side by side
    combined_width = annotated.width + text_panel_width
    combined = Image.new('RGB', (combined_width, annotated.height), (255, 255, 255))
    combined.paste(annotated, (0, 0))
    combined.paste(text_panel, (annotated.width, 0))

    return combined


def fetch_paper_data(db: DatabaseManager, paper_id: str) -> Optional[dict]:
    """Fetch PDF content and OCR results from database."""
    with db.session_scope() as session:
        paper = session.query(ArxivPaper).filter(ArxivPaper.id == paper_id).first()
        if not paper:
            return None

        return {
            'id': paper.id,
            'title': paper.title,
            'pdf_content': paper.pdf_content,
            'ocr_results': paper.ocr_results,
            'ocr_processed': paper.ocr_processed,
        }


def list_available_papers(db: DatabaseManager, limit: int = 20) -> List[dict]:
    """List papers that have OCR results available."""
    with db.session_scope() as session:
        papers = session.query(
            ArxivPaper.id,
            ArxivPaper.title,
            ArxivPaper.ocr_processed,
        ).filter(
            ArxivPaper.ocr_results.isnot(None),
        ).order_by(ArxivPaper.id).limit(limit).all()

        return [{'id': p.id, 'title': p.title[:60], 'processed': p.ocr_processed} for p in papers]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize OCR results from database with bounding boxes and text"
    )
    parser.add_argument(
        "--paper-id",
        help="Paper ID to visualize (e.g., '0704.0001')"
    )
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="Page index to visualize (0-based, default: 0)"
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Visualize all pages"
    )
    parser.add_argument(
        "--output",
        help="Output image path (default: display info, use --save to save)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save visualization to outputs/visualizations/"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available papers with OCR results"
    )
    parser.add_argument(
        "--text-width",
        type=int,
        default=600,
        help="Width of text panel in pixels (default: 600)"
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=16,
        help="Font size for text panel (default: 16)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="DPI for PDF rendering (default: auto-detect from OCR results)"
    )
    # Database args
    parser.add_argument("--db-url", default=os.getenv("SQLALCHEMY_URL"), help="DB URL")
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", ""), help="DB host")
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")))
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "arxiv"))
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"))
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD", ""))

    args = parser.parse_args()

    # Connect to database
    db = DatabaseManager(
        db_url=args.db_url,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
    )
    db.create_engine_and_session()

    # List mode
    if args.list:
        print("\nPapers with OCR results:")
        print("-" * 80)
        papers = list_available_papers(db, limit=30)
        if not papers:
            print("No papers with OCR results found.")
        else:
            for p in papers:
                status = "✓" if p['processed'] else "⋯"
                print(f"  {status} {p['id']:15} {p['title']}")
        print("-" * 80)
        print(f"Total: {len(papers)} papers")
        print("\nUse --paper-id <ID> to visualize a specific paper")
        return

    # Require paper ID for visualization
    if not args.paper_id:
        print("Error: --paper-id is required (use --list to see available papers)")
        sys.exit(1)

    if not HAS_PDF2IMAGE:
        print("Error: pdf2image is required. Install with: pip install pdf2image")
        sys.exit(1)

    # Fetch paper data
    print(f"\nFetching paper: {args.paper_id}")
    paper_data = fetch_paper_data(db, args.paper_id)

    if not paper_data:
        print(f"Error: Paper '{args.paper_id}' not found in database")
        sys.exit(1)

    if not paper_data['pdf_content']:
        print(f"Error: Paper '{args.paper_id}' has no PDF content")
        sys.exit(1)

    if not paper_data['ocr_results']:
        print(f"Error: Paper '{args.paper_id}' has no OCR results")
        sys.exit(1)

    print(f"Title: {paper_data['title'][:80]}...")

    ocr_data = paper_data['ocr_results']
    results = ocr_data.get('results', [])
    total_pages = ocr_data.get('total_pages', len(results))

    print(f"Total pages: {total_pages}, OCR processed: {len(results)} pages")

    # Determine DPI
    if args.dpi:
        rendering_dpi = args.dpi
    elif results and 'dpi' in results[0]:
        rendering_dpi = int(results[0]['dpi'])
    else:
        rendering_dpi = 200  # Default
    print(f"Rendering DPI: {rendering_dpi}")

    # Determine pages to process
    if args.all_pages:
        pages_to_process = list(range(len(results)))
    else:
        if args.page >= len(results):
            print(f"Error: Page {args.page} not available (OCR has {len(results)} pages)")
            sys.exit(1)
        pages_to_process = [args.page]

    # Process pages
    pdf_bytes = paper_data['pdf_content']

    for page_idx in pages_to_process:
        print(f"\nProcessing page {page_idx + 1}/{len(results)}...")

        # Convert PDF page to image
        try:
            images = convert_from_bytes(
                pdf_bytes,
                dpi=rendering_dpi,
                first_page=page_idx + 1,
                last_page=page_idx + 1,
            )
        except Exception as e:
            print(f"Error rendering page {page_idx}: {e}")
            continue

        if not images:
            print(f"Warning: Could not render page {page_idx}")
            continue

        page_image = images[0]

        # Get OCR boxes for this page
        page_result = results[page_idx]
        layout_det = page_result.get('layout_det_res', {})
        boxes = layout_det.get('boxes', [])

        print(f"  Image size: {page_image.width}x{page_image.height}")
        print(f"  Found {len(boxes)} layout boxes")

        # Create visualization
        vis_image = visualize_page(
            page_image,
            boxes,
            text_panel_width=args.text_width,
            font_size=args.font_size,
        )

        # Save or display path
        if args.output and not args.all_pages:
            output_path = Path(args.output)
        elif args.save or args.output:
            output_dir = Path("outputs/visualizations")
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_id = args.paper_id.replace('/', '_')
            output_path = output_dir / f"{safe_id}_page_{page_idx}.png"
        else:
            # Just show info, don't save
            print(f"  Visualization ready ({vis_image.width}x{vis_image.height})")
            print(f"  Use --save to save to file, or --output <path> to specify path")
            continue

        vis_image.save(output_path, quality=95)
        print(f"  Saved: {output_path}")

        # Clean up
        page_image.close()
        for img in images:
            img.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
