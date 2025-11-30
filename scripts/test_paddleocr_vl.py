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
from typing import Optional, Any
from dotenv import load_dotenv

# Add project root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(override=True)

try:
    import paddle
    from paddleocr import PaddleOCRVL  # type: ignore
except ImportError:
    raise SystemExit(
        "paddleocr is not installed. Install paddlepaddle[-gpu] and paddleocr to use this script."
    )


def run_ocr(pdf_path: Path, pages: int, device: str, model_dir: Path, output: Path):
    print(f"Running PaddleOCR-VL on: {pdf_path}")
    print(f"Pages: {pages} | Device: {device}")
    print(f"Model dir: {model_dir}")

    try:
        paddle.device.set_device(device)
    except Exception as e:
        print(f"Warning: could not set device '{device}', falling back to default: {e}")

    try:
        ocr = PaddleOCRVL(
            use_layout_detection=True,
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            layout_detection_model_dir=str(model_dir / "PP-DocLayoutV2"),
            vl_rec_model_dir=str(model_dir),
        )
    except Exception as e:
        raise SystemExit(
            f"Failed to initialize PaddleOCR-VL (likely missing model files or device not available): {e}"
        )

    results = ocr.predict(str(pdf_path))
    if isinstance(results, list) and pages > 0:
        results = results[:pages]

    def to_serializable(obj: Any):
        try:
            import numpy as np  # type: ignore
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
        except Exception:
            pass
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        # Fallback: stringify non-JSON-serializable objects (e.g., Font, Paddle classes)
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"

    results = to_serializable(results)

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
        "--device",
        default="gpu:0",
        help="Device to use (e.g., gpu:0 or cpu)",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/.paddlex/official_models"),
        help="Directory containing PaddleOCR-VL model files",
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
    model_dir = Path(args.model_dir)

    run_ocr(pdf_path=pdf_path, pages=args.pages, device=args.device, model_dir=model_dir, output=output)


if __name__ == "__main__":
    main()
