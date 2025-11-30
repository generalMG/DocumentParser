#!/usr/bin/env python3
"""
Experimental PaddleOCR-VL test script for PDF analysis.

This is optional and requires extra dependencies:
    - paddlepaddle-gpu (or paddlepaddle for CPU)
    - paddleocr

It runs OCR on the first N pages of a PDF and saves raw results to JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Add project root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(override=True)

try:
    from paddleocr import PaddleOCR  # type: ignore
except ImportError:
    raise SystemExit(
        "paddleocr is not installed. Install paddlepaddle[-gpu] and paddleocr to use this script."
    )


def run_ocr(pdf_path: Path, pages: int, use_gpu: bool, lang: str, output: Path):
    print(f"Running PaddleOCR-VL on: {pdf_path}")
    print(f"Pages: {pages} | GPU: {use_gpu} | Lang: {lang}")

    ocr = PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        use_gpu=use_gpu,
        ocr_version="PP-OCRv4",
        enable_mkldnn=not use_gpu,
    )

    results = ocr.ocr(str(pdf_path), cls=True, page_num=pages)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved OCR results to {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test PaddleOCR-VL on a PDF (experimental)"
    )
    parser.add_argument(
        "pdf",
        help="Path to PDF file",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of pages to process (from start)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code for OCR",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (default uses GPU if available)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file (default: outputs/ocr_<pdf-stem>.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    output = Path(args.output) if args.output else Path("outputs") / f"ocr_{pdf_path.stem}.json"
    use_gpu = not args.cpu

    run_ocr(pdf_path=pdf_path, pages=args.pages, use_gpu=use_gpu, lang=args.lang, output=output)


if __name__ == "__main__":
    main()
