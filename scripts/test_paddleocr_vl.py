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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple
import multiprocessing as mp
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

try:
    from safetensors import safe_open as original_safe_open
    import torch
    import numpy as np

    def torch_to_paddle(pt_tensor):
        # Convert torch tensor to paddle tensor
        # Handle bfloat16 specifically because numpy doesn't support it
        if pt_tensor.dtype == torch.bfloat16:
            np_array = pt_tensor.float().numpy()
            pd_tensor = paddle.to_tensor(np_array)
            return pd_tensor.cast("bfloat16")
        else:
            np_array = pt_tensor.numpy()
            return paddle.to_tensor(np_array)

    class SafeSliceWrapper:
        def __init__(self, slice_obj):
            self.slice_obj = slice_obj

        def __getitem__(self, item):
            # Get torch tensor slice and convert to paddle tensor
            pt_tensor = self.slice_obj[item]
            return torch_to_paddle(pt_tensor)

        def get_shape(self):
            return self.slice_obj.get_shape()

    class SafeOpenWrapper:
        def __init__(self, path, framework, device):
            self.path = path
            self.framework = framework
            self.device = device
            self.f = None

        def __enter__(self):
            # Force framework="pt" to handle bfloat16
            self.f = original_safe_open(self.path, framework="pt", device=self.device)
            self.inner = self.f.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self.f.__exit__(exc_type, exc_value, traceback)

        def get_tensor(self, key):
            # Get torch tensor and convert to paddle tensor
            pt_tensor = self.inner.get_tensor(key)
            return torch_to_paddle(pt_tensor)

        def get_slice(self, key):
            return SafeSliceWrapper(self.inner.get_slice(key))

        def keys(self):
            return self.inner.keys()
        
        def metadata(self):
            return self.inner.metadata()

    def monkeypatch_safe_open(path, framework="pt", device="cpu"):
        if framework == "paddle":
            # Use PyTorch as backend to read tensors (supports bfloat16)
            return SafeOpenWrapper(path, framework="pt", device=device)
        return original_safe_open(path, framework=framework, device=device)

    import safetensors
    safetensors.safe_open = monkeypatch_safe_open
    print("Monkeypatched safetensors.safe_open to support 'paddle' framework via torch.")

except ImportError:
    pass  # safetensors or torch might not be installed



def _to_serializable(obj: Any):
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def _process_pages(
    pdf_path: str,
    device: str,
    model_dir: str,
    page_indices: List[int],
) -> List[Tuple[int, Any]]:
    """Worker to process a subset of pages."""
    import paddle
    from paddleocr import PaddleOCRVL  # type: ignore

    try:
        paddle.device.set_device(device)
    except Exception:
        pass

    ocr = PaddleOCRVL(
        use_layout_detection=True,
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        layout_detection_model_dir=os.path.join(model_dir, "PP-DocLayoutV2"),
        vl_rec_model_dir=model_dir,
    )

    results = ocr.predict(pdf_path)
    # select requested pages if available
    out: List[Tuple[int, Any]] = []
    for idx in page_indices:
        if isinstance(results, list) and idx < len(results):
            out.append((idx, results[idx]))
    return out


def run_ocr(pdf_path: Path, pages: int, device: str, model_dir: Path, output: Path, processes: int):
    print(f"Running PaddleOCR-VL on: {pdf_path}")
    print(f"Pages: {pages} | Device: {device}")
    print(f"Model dir: {model_dir}")
    print(f"Processes: {processes}")

    if pages < 1:
        raise SystemExit("pages must be >= 1")

    if processes <= 1:
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
    else:
        page_indices = list(range(pages))
        # split indices across workers
        chunks = [page_indices[i::processes] for i in range(processes)]
        collected: Dict[int, Any] = {}
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=processes, mp_context=ctx) as ex:
            futures = [
                ex.submit(
                    _process_pages,
                    str(pdf_path),
                    device,
                    str(model_dir),
                    chunk,
                )
                for chunk in chunks
                if chunk
            ]
            for fut in as_completed(futures):
                for idx, res in fut.result():
                    collected[idx] = res
        results = [collected[i] for i in sorted(collected.keys()) if i < pages]

    results = _to_serializable(results)

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
        "--processes",
        type=int,
        default=1,
        help="Number of parallel processes (each loads its own model)",
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

    run_ocr(
        pdf_path=pdf_path,
        pages=args.pages,
        device=args.device,
        model_dir=model_dir,
        output=output,
        processes=args.processes,
    )


if __name__ == "__main__":
    main()
