#!/usr/bin/env python3
"""
Visualize OCR results with bounding boxes and labels.
Similar to PaddleOCR's visualization style.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import sys

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False
    print("Warning: pdf2image not installed. Install with: pip install pdf2image")
    print("Also requires poppler: sudo apt-get install poppler-utils (Linux) or brew install poppler (Mac)")

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print("Error: PIL/Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


def get_color_for_label(label: str) -> Tuple[int, int, int]:
    """Get a consistent color for each label type."""
    colors = {
        'doc_title': (255, 0, 0),      # Red
        'header': (0, 255, 0),          # Green
        'footer': (0, 0, 255),          # Blue
        'text': (255, 165, 0),          # Orange
        'aside_text': (128, 0, 128),    # Purple
        'table': (0, 255, 255),         # Cyan
        'figure': (255, 192, 203),      # Pink
        'figure_caption': (165, 42, 42),# Brown
        'table_caption': (0, 128, 128), # Teal
        'equation': (255, 255, 0),      # Yellow
        'list': (128, 128, 0),          # Olive
        'code': (0, 128, 0),            # Dark Green
    }
    return colors.get(label, (128, 128, 128))  # Default gray


def draw_bbox_with_label(
    draw: ImageDraw.ImageDraw,
    bbox: List[float],
    label: str,
    score: float,
    line_width: int = 2,
    font_size: int = 12,
):
    """Draw a bounding box with label on top left."""
    x1, y1, x2, y2 = bbox

    # Use coordinates directly - PaddleOCR returns pixel coordinates
    # at the resolution it processed the PDF at (~140 DPI)
    x1_scaled = x1
    y1_scaled = y1
    x2_scaled = x2
    y2_scaled = y2

    # Get color for this label
    color = get_color_for_label(label)

    # Draw rectangle
    draw.rectangle(
        [(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)],
        outline=color,
        width=line_width
    )

    # Prepare label text
    label_text = f"{label} ({score:.2f})"

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()

    # Get text bounding box
    bbox_text = draw.textbbox((0, 0), label_text, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]

    # Draw label background (above the box)
    label_y = max(0, y1_scaled - text_height - 4)
    draw.rectangle(
        [(x1_scaled, label_y), (x1_scaled + text_width + 4, label_y + text_height + 4)],
        fill=color
    )

    # Draw label text
    draw.text(
        (x1_scaled + 2, label_y + 2),
        label_text,
        fill=(255, 255, 255),
        font=font
    )


def visualize_page(
    image: Image.Image,
    boxes: List[dict],
    show_content: bool = False,
) -> Image.Image:
    """Visualize OCR results on a page image."""
    # Create a copy to draw on
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)

    # Draw all bounding boxes
    for box in boxes:
        draw_bbox_with_label(
            draw,
            box['coordinate'],
            box['label'],
            float(box['score']),
            line_width=2,
            font_size=12,
        )

    # Optionally create a side-by-side view with text content
    if show_content:
        # Create text panel
        text_width = 400
        text_image = Image.new('RGB', (text_width, vis_image.height), (255, 255, 255))
        text_draw = ImageDraw.Draw(text_image)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()

        y_offset = 10
        for i, box in enumerate(boxes):
            content = box.get('content', '')[:100]  # Truncate long content
            text = f"{i+1}. [{box['label']}] {content}"

            # Word wrap
            lines = []
            words = text.split()
            current_line = []
            for word in words:
                test_line = ' '.join(current_line + [word])
                bbox = text_draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] < text_width - 20:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))

            # Draw lines
            for line in lines:
                if y_offset < text_image.height - 20:
                    text_draw.text((10, y_offset), line, fill=(0, 0, 0), font=font)
                    y_offset += 15
            y_offset += 5

        # Combine images side by side
        combined = Image.new('RGB', (vis_image.width + text_width, vis_image.height))
        combined.paste(vis_image, (0, 0))
        combined.paste(text_image, (vis_image.width, 0))
        return combined

    return vis_image


def get_paddleocr_dpi(
    pdf_path: Path,
    json_data: dict,
    page_idx: int = 0,
) -> int:
    """Infer the DPI that PaddleOCR used to process the PDF.

    PaddleOCR returns coordinates in pixel space at the resolution it used.
    This function infers that DPI from the coordinate ranges.
    """
    # Check if DPI is explicitly provided in the results (new standard)
    try:
        page_data = json_data['results'][page_idx]
        if 'dpi' in page_data:
            return int(page_data['dpi'])
    except (KeyError, IndexError, TypeError):
        pass

    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("Warning: PyPDF2 not available, using default 144 DPI")
        return 144

    # Get PDF page dimensions in points
    pdf_reader = PdfReader(pdf_path)
    page = pdf_reader.pages[page_idx]
    pdf_width = float(page.mediabox.width)
    pdf_height = float(page.mediabox.height)

    # Get coordinate ranges from JSON
    page_data = json_data['results'][page_idx]
    boxes = page_data['layout_det_res']['boxes']

    if not boxes:
        return 144

    max_x = max(box['coordinate'][2] for box in boxes)
    max_y = max(box['coordinate'][3] for box in boxes)

    # Infer the DPI that PaddleOCR used
    # DPI = (pixels / points) * 72
    # These are lower bounds because text might not reach the edge
    raw_dpi_x = (max_x / pdf_width) * 72
    raw_dpi_y = (max_y / pdf_height) * 72
    
    # The true DPI must be at least the max of these (assuming content fits on page)
    min_dpi = max(raw_dpi_x, raw_dpi_y)
    
    # Standard DPIs used by OCR engines and PDF renderers
    # 144 is common (2x 72), 200 is default for pdf2image, 96/150/300 are also common
    standard_dpis = [72, 96, 100, 144, 150, 200, 300]
    
    # Find the smallest standard DPI that fits the content
    # Allow a small tolerance (1%) for floating point errors
    for dpi in standard_dpis:
        if dpi >= min_dpi * 0.99:
            return dpi
            
    # If no standard DPI fits (e.g. very high res), return the raw calculated DPI
    # rounded to nearest integer
    return int(min_dpi)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize OCR results with bounding boxes and labels"
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to OCR results JSON file"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to source PDF file"
    )
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="Page index to visualize (0-based, default: 0)"
    )
    parser.add_argument(
        "--output",
        help="Output image path (default: outputs/vis_page_<N>.png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="DPI for PDF rendering (default: auto-detect from JSON, typically ~140)"
    )
    parser.add_argument(
        "--show-content",
        action="store_true",
        help="Show text content in side panel"
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Visualize all pages"
    )

    args = parser.parse_args()

    # Load JSON
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    # Check PDF
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    if not HAS_PDF2IMAGE:
        print("Error: pdf2image is required for visualization")
        print("Install with: pip install pdf2image")
        print("Also requires poppler-utils")
        sys.exit(1)

    # Determine pages to process
    total_pages = len(data.get('results', []))
    if args.all_pages:
        pages_to_process = list(range(total_pages))
    else:
        if args.page >= total_pages:
            print(f"Error: Page {args.page} out of range (total pages: {total_pages})")
            sys.exit(1)
        pages_to_process = [args.page]

    # Determine DPI to use for rendering
    if args.dpi is None:
        # Auto-detect from JSON
        paddleocr_dpi = get_paddleocr_dpi(pdf_path, data, page_idx=0)
        rendering_dpi = paddleocr_dpi
        print(f"Auto-detected PaddleOCR DPI: {paddleocr_dpi}")
    else:
        rendering_dpi = args.dpi
        paddleocr_dpi = get_paddleocr_dpi(pdf_path, data, page_idx=0)
        print(f"User-specified DPI: {rendering_dpi}")
        print(f"Warning: PaddleOCR used {paddleocr_dpi} DPI, coordinates may not align correctly")

    # Process pages
    for page_idx in pages_to_process:
        print(f"\nProcessing page {page_idx}...")

        # Render PDF page at the same DPI PaddleOCR used
        images = convert_from_path(
            pdf_path,
            dpi=rendering_dpi,
            first_page=page_idx + 1,
            last_page=page_idx + 1
        )

        if not images:
            print(f"Warning: Could not render page {page_idx}")
            continue

        page_image = images[0]

        # Get boxes for this page
        page_data = data['results'][page_idx]
        boxes = page_data['layout_det_res']['boxes']

        print(f"  Found {len(boxes)} boxes")

        # Visualize
        vis_image = visualize_page(
            page_image,
            boxes,
            show_content=args.show_content,
        )

        # Save
        if args.output and not args.all_pages:
            output_path = Path(args.output)
        else:
            output_dir = Path("outputs/visualizations")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"vis_page_{page_idx}.png"

        vis_image.save(output_path)
        print(f"  Saved: {output_path}")

    print(f"\nVisualization complete!")


if __name__ == "__main__":
    main()
