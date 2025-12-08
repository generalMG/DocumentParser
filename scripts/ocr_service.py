#!/usr/bin/env python3
"""
FastAPI service that hosts PaddleOCR-VL and processes uploaded PDFs.

- Keeps a pool of PaddleOCR-VL instances (one per process).
- Accepts PDF uploads and returns OCR results as JSON.
- Model paths are read from a local directory (pre-downloaded).
"""

import argparse
import asyncio
import gc
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from typing import Any, List, Optional
import traceback
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from safetensors import safe_open as original_safe_open
    import torch
    import numpy as np
    import paddle

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


app = FastAPI(title="PaddleOCR-VL Service")

POOL: Optional[ProcessPoolExecutor] = None
SERVICE_ARGS = None


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and log them."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


def _filter_image_data(obj: Any, depth: int = 0) -> Any:
    """Remove image data and other large binary data from results, keeping only text/metadata."""
    # Prevent infinite recursion
    if depth > 50:
        return "<max_depth_reached>"

    # Skip image-related fields completely
    if isinstance(obj, dict):
        filtered = {}
        for k, v in obj.items():
            # Skip fields that contain image data
            if k in ['input_img', 'img', 'image', 'visualization', 'vis_img', 'preprocessed_img']:
                logger.debug(f"Skipping image field: {k}")
                continue
            # Skip large numpy arrays (likely images)
            try:
                import numpy as np
                if isinstance(v, np.ndarray) and v.size > 1000:  # Skip large arrays
                    logger.debug(f"Skipping large array: {k} (size: {v.size})")
                    continue
            except:
                pass
            filtered[k] = _filter_image_data(v, depth + 1)
        return filtered
    elif isinstance(obj, (list, tuple)):
        # Skip if this looks like image data (list of pixel values)
        if len(obj) > 100 and isinstance(obj[0], (list, tuple)):
            # This might be image data, skip it
            try:
                if isinstance(obj[0][0], (int, float)):
                    logger.debug(f"Skipping suspected image array (length: {len(obj)})")
                    return None
            except:
                pass
        filtered_list = []
        for x in obj:
            filtered_item = _filter_image_data(x, depth + 1)
            if filtered_item is not None:
                filtered_list.append(filtered_item)
        return filtered_list
    else:
        return obj


def _to_serializable(obj: Any):
    """Convert to JSON-serializable format after filtering image data."""
    # First filter out image data
    obj = _filter_image_data(obj)

    # Then convert to serializable format
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.ndarray):
            # Only convert small arrays (likely coordinates/bboxes, not images)
            if obj.size > 1000:
                return None  # Skip large arrays
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, (list, tuple)):
        filtered = [_to_serializable(x) for x in obj]
        return [x for x in filtered if x is not None]  # Remove None values
    if isinstance(obj, dict):
        filtered = {k: _to_serializable(v) for k, v in obj.items()}
        return {k: v for k, v in filtered.items() if v is not None}  # Remove None values
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def _check_gpu_availability(device: str) -> dict:
    """Check GPU availability and memory."""
    try:
        import paddle

        if device.startswith("gpu"):
            gpu_id = int(device.split(":")[-1]) if ":" in device else 0

            # Check if GPU is available
            if not paddle.device.is_compiled_with_cuda():
                return {"ok": False, "error": "Paddle not compiled with CUDA"}

            gpu_count = paddle.device.cuda.device_count()
            if gpu_id >= gpu_count:
                return {"ok": False, "error": f"GPU {gpu_id} not available (only {gpu_count} GPUs found)"}

            # Get GPU memory info
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                pynvml.nvmlShutdown()

                return {
                    "ok": True,
                    "device": device,
                    "gpu_name": gpu_name,
                    "total_memory_gb": mem_info.total / 1024**3,
                    "free_memory_gb": mem_info.free / 1024**3,
                    "used_memory_gb": mem_info.used / 1024**3,
                }
            except Exception:
                # Fallback if pynvml not available
                return {"ok": True, "device": device, "gpu_id": gpu_id}
        else:
            return {"ok": True, "device": "cpu"}
    except Exception as e:
        return {"ok": False, "error": f"GPU check failed: {e}"}


def _init_model_worker(device: str, model_dir: str) -> dict:
    """Initialize model in worker process and run warm-up."""
    try:
        import paddle
        from paddleocr import PaddleOCRVL  # type: ignore
        from io import BytesIO
        from reportlab.pdfgen import canvas
    except Exception as e:
        return {"ok": False, "error": f"Import failed: {e}"}

    global _MODEL

    # Set device
    try:
        paddle.device.set_device(device)
    except Exception as e:
        return {"ok": False, "error": f"Failed to set device {device}: {e}"}

    # Check GPU before loading
    gpu_check = _check_gpu_availability(device)
    if not gpu_check["ok"]:
        return gpu_check

    print(f"Worker initializing on {device}...")
    if device.startswith("gpu") and "gpu_name" in gpu_check:
        print(f"  GPU: {gpu_check['gpu_name']}")
        print(f"  Memory: {gpu_check['free_memory_gb']:.2f} GB free / {gpu_check['total_memory_gb']:.2f} GB total")

    # Load model
    try:
        _MODEL = PaddleOCRVL(
            use_layout_detection=True,
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            layout_detection_model_dir=os.path.join(model_dir, "PP-DocLayoutV2"),
            vl_rec_model_dir=model_dir,
        )
        print(f"  Model loaded successfully")
    except Exception as e:
        return {"ok": False, "error": f"Model loading failed: {e}"}

    # Create a minimal dummy PDF for warm-up
    try:
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer)
        c.drawString(100, 750, "Warm-up test")
        c.save()
        pdf_buffer.seek(0)
        dummy_pdf = pdf_buffer.getvalue()

        # Run warm-up inference
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(dummy_pdf)
            tmp.flush()
            _ = _MODEL.predict(tmp.name)

        print(f"  Warm-up completed successfully")
    except Exception as e:
        print(f"  Warning: Warm-up failed: {e}")

    # Check GPU memory after loading
    if device.startswith("gpu"):
        gpu_check_after = _check_gpu_availability(device)
        if gpu_check_after["ok"] and "used_memory_gb" in gpu_check_after:
            print(f"  GPU Memory after load: {gpu_check_after['used_memory_gb']:.2f} GB used")

    return {"ok": True, "device": device}


def _parse_parsing_res(parsing_res_item) -> dict:
    """Parse a single item from parsing_res_list to extract label, bbox, and content."""
    try:
        # Check if it's a PaddleOCRVLBlock object (has attributes)
        if hasattr(parsing_res_item, 'label') and hasattr(parsing_res_item, 'bbox'):
            # Extract directly from object attributes
            parsed = {}
            if hasattr(parsing_res_item, 'label'):
                parsed['label'] = str(parsing_res_item.label)
            if hasattr(parsing_res_item, 'bbox'):
                parsed['bbox'] = parsing_res_item.bbox
            if hasattr(parsing_res_item, 'content'):
                parsed['content'] = str(parsing_res_item.content)
            elif hasattr(parsing_res_item, 'text'):
                parsed['content'] = str(parsing_res_item.text)
            return parsed

        # Otherwise, try to parse as string (for backwards compatibility)
        if isinstance(parsing_res_item, str):
            lines = parsing_res_item.strip().split('\n')
            parsed = {}
            for line in lines:
                line = line.strip()
                if line.startswith('label:'):
                    parsed['label'] = line.replace('label:', '').strip()
                elif line.startswith('bbox:'):
                    bbox_str = line.replace('bbox:', '').strip()
                    # Parse [x1, y1, x2, y2]
                    bbox_str = bbox_str.strip('[]')
                    parsed['bbox'] = [float(x.strip()) for x in bbox_str.split(',')]
                elif line.startswith('content:'):
                    parsed['content'] = line.replace('content:', '').strip()
            return parsed

        return {}
    except Exception as e:
        logger.warning(f"Failed to parse parsing_res item: {e}")
        return {}


def _bbox_iou(bbox1: list, bbox2: list) -> float:
    """Calculate Intersection over Union (IoU) between two bboxes [x1, y1, x2, y2]."""
    try:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
    except (IndexError, TypeError):
        return 0.0


def _enrich_results(results: Any) -> Any:
    """Enrich layout_det_res boxes with content from parsing_res_list by matching bounding boxes."""
    if not isinstance(results, list):
        return results

    enriched = []
    for page_result in results:
        if not isinstance(page_result, dict):
            enriched.append(page_result)
            continue

        # Parse parsing_res_list
        parsing_list = page_result.get('parsing_res_list', [])
        parsed_items = [_parse_parsing_res(item) for item in parsing_list]

        # Enrich layout_det_res boxes
        layout_det_res = page_result.get('layout_det_res', {})
        boxes = layout_det_res.get('boxes', [])

        # Match boxes to parsed items by bounding box IoU
        for box in boxes:
            box_coord = box.get('coordinate', [])
            if len(box_coord) < 4:
                continue

            best_match = None
            best_iou = 0.0

            for parsed in parsed_items:
                parsed_bbox = parsed.get('bbox', [])
                if len(parsed_bbox) < 4:
                    continue

                iou = _bbox_iou(box_coord, parsed_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = parsed

            # Require reasonable overlap (IoU > 0.5) to match
            if best_match and best_iou > 0.5:
                box['content'] = best_match.get('content', '')
            elif best_match and best_iou > 0.1:
                # Lower threshold fallback - still assign if some overlap
                box['content'] = best_match.get('content', '')

        enriched.append(page_result)

    return enriched


def _cleanup_gpu_memory():
    """Clean up GPU memory and trigger garbage collection."""
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception as e:
        logger.debug(f"GPU cache clear failed: {e}")

    # Force garbage collection
    gc.collect()


def _process_pdf(pdf_bytes: bytes, pages: int, device: str, model_dir: str) -> dict:
    """Worker: run predict on PDF. Model must be pre-loaded. Returns dict to avoid pickling errors."""
    global _MODEL

    # Initialize model if not already loaded (fallback for legacy code paths)
    if "_MODEL" not in globals() or _MODEL is None:
        init_result = _init_model_worker(device, model_dir)
        if not init_result["ok"]:
            return init_result

    images = None
    try:
        from pdf2image import convert_from_bytes

        # Convert PDF to images at fixed DPI
        # This ensures consistent coordinates regardless of PDF dimensions
        target_dpi = 200
        images = convert_from_bytes(pdf_bytes, dpi=target_dpi)

        if pages and pages > 0:
            images = images[:pages]

        results = []
        for i, img in enumerate(images):
            # Save to temp file for PaddleOCR (it expects file path or numpy array)
            # Using file path is safer for memory with large images
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_img:
                img.save(tmp_img.name)
                page_result = _MODEL.predict(tmp_img.name)

                # If result is a list (multi-page), take the first item since we process one by one
                if isinstance(page_result, list) and len(page_result) > 0:
                    page_result = page_result[0]

                # Add page index
                if isinstance(page_result, dict):
                    page_result['page_index'] = str(i)

                results.append(page_result)

            # Close PIL image after processing each page
            img.close()

        # Enrich results with content in layout boxes
        results = _enrich_results(results)

        # Inject DPI info into results
        serializable_results = _to_serializable(results)
        if isinstance(serializable_results, list):
            for res in serializable_results:
                if isinstance(res, dict):
                    res['dpi'] = target_dpi

        return {"ok": True, "results": serializable_results}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        # Clean up PIL images
        if images:
            for img in images:
                try:
                    img.close()
                except Exception:
                    pass
            del images

        # Clean up GPU memory
        _cleanup_gpu_memory()


def _process_pdf_page(pdf_bytes: bytes, page_num: int, device: str, model_dir: str) -> dict:
    """Worker: process a single page of a PDF. Model must be pre-loaded. Returns dict to avoid pickling errors."""
    global _MODEL

    # Initialize model if not already loaded (fallback for legacy code paths)
    if "_MODEL" not in globals() or _MODEL is None:
        init_result = _init_model_worker(device, model_dir)
        if not init_result["ok"]:
            return init_result

    images = None
    image = None
    pdf_reader = None
    try:
        from pdf2image import convert_from_bytes
        from PyPDF2 import PdfReader
        from io import BytesIO

        # Get total pages first (before heavy image conversion)
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)

        # Convert specific page to image at fixed DPI
        target_dpi = 200

        # pdf2image uses 1-based indexing for first_page/last_page
        images = convert_from_bytes(
            pdf_bytes,
            dpi=target_dpi,
            first_page=page_num + 1,
            last_page=page_num + 1
        )

        if not images:
            return {"ok": False, "error": f"Page {page_num} could not be rendered"}

        image = images[0]

        # Process the single page image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            image.save(tmp.name)
            results = _MODEL.predict(tmp.name)

        # Enrich results with content in layout boxes
        results = _enrich_results(results)

        # Results should be a list with one element (the single page)
        if isinstance(results, list) and len(results) > 0:
            result = _to_serializable(results[0])
            if isinstance(result, dict):
                result['dpi'] = target_dpi

            return {"ok": True, "result": result, "total_pages": total_pages}
        else:
            return {"ok": False, "error": "Unexpected results format from single page"}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return {"ok": False, "error": f"{str(e)}\n{error_trace}"}
    finally:
        # Clean up PIL image
        if image:
            try:
                image.close()
            except Exception:
                pass
            del image

        # Clean up images list
        if images:
            for img in images:
                try:
                    img.close()
                except Exception:
                    pass
            del images

        # Clean up PDF reader
        if pdf_reader:
            try:
                pdf_reader.stream.close()
            except Exception:
                pass
            del pdf_reader

        # Clean up GPU memory after each page
        _cleanup_gpu_memory()


@app.on_event("startup")
async def _startup():
    global POOL
    if SERVICE_ARGS is None:
        return
    # fall back to threads if process pool creation fails (e.g., permission errors)
    try:
        ctx = mp.get_context("spawn")
        POOL = ProcessPoolExecutor(max_workers=SERVICE_ARGS.workers, mp_context=ctx)
    except Exception as e:
        print(f"Process pool unavailable ({e}), falling back to ThreadPoolExecutor")
        from concurrent.futures import ThreadPoolExecutor
        POOL = ThreadPoolExecutor(max_workers=SERVICE_ARGS.workers)

    # Pre-initialize all workers with model loading and warm-up
    print(f"\nInitializing {SERVICE_ARGS.workers} worker(s) with device={SERVICE_ARGS.device}...")
    loop = asyncio.get_running_loop()
    init_tasks = []
    for i in range(SERVICE_ARGS.workers):
        init_tasks.append(
            loop.run_in_executor(
                POOL, _init_model_worker, SERVICE_ARGS.device, SERVICE_ARGS.model_dir
            )
        )

    # Wait for all workers to initialize
    try:
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Worker {i+1} initialization failed: {result}")
                raise RuntimeError(f"Worker initialization failed: {result}")
            elif not result.get("ok"):
                print(f"Worker {i+1} initialization failed: {result.get('error')}")
                raise RuntimeError(f"Worker initialization failed: {result.get('error')}")
            else:
                print(f"Worker {i+1} ready on {result.get('device')}")
        print(f"\nAll {SERVICE_ARGS.workers} worker(s) initialized successfully!\n")
    except Exception as e:
        print(f"\nFATAL: Worker initialization failed: {e}")
        if POOL:
            POOL.shutdown(wait=False)
            POOL = None
        raise


@app.on_event("shutdown")
async def _shutdown():
    global POOL
    if POOL:
        POOL.shutdown(wait=True)
        POOL = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    pages: int = Form(None),
):
    logger.info(f"OCR request received: filename={file.filename}, pages={pages}")

    if POOL is None:
        logger.error("Worker pool not ready")
        raise HTTPException(status_code=503, detail="Worker pool not ready")

    try:
        data = await file.read()
        logger.info(f"Read {len(data)} bytes from {file.filename}")
    except Exception as e:
        logger.error(f"Failed to read file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    loop = asyncio.get_running_loop()
    try:
        logger.info(f"Submitting OCR task for {file.filename}")
        result = await loop.run_in_executor(
            POOL, _process_pdf, data, pages, SERVICE_ARGS.device, SERVICE_ARGS.model_dir
        )
        logger.info(f"OCR task completed for {file.filename}")
    except Exception as e:
        logger.error(f"OCR execution failed for {file.filename}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    if not result.get("ok"):
        error_msg = result.get('error', 'Unknown error')
        logger.error(f"OCR processing failed for {file.filename}: {error_msg}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {error_msg}")

    logger.info(f"OCR successful for {file.filename}")
    return JSONResponse({"results": result.get("results")})


@app.post("/ocr_page")
async def ocr_page_endpoint(
    file: UploadFile = File(...),
    page: int = Form(...),
):
    logger.info(f"OCR page request received: filename={file.filename}, page={page}")

    if POOL is None:
        logger.error("Worker pool not ready")
        raise HTTPException(status_code=503, detail="Worker pool not ready")

    try:
        data = await file.read()
        logger.info(f"Read {len(data)} bytes from {file.filename} for page {page}")
    except Exception as e:
        logger.error(f"Failed to read file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    loop = asyncio.get_running_loop()
    try:
        logger.info(f"Submitting OCR page task for {file.filename} page {page}")
        result = await loop.run_in_executor(
            POOL, _process_pdf_page, data, page, SERVICE_ARGS.device, SERVICE_ARGS.model_dir
        )
        logger.info(f"OCR page task completed for {file.filename} page {page}")
    except Exception as e:
        logger.error(f"OCR page execution failed for {file.filename} page {page}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    if not result.get("ok"):
        error_msg = result.get('error', 'Unknown error')
        logger.error(f"OCR page processing failed for {file.filename} page {page}: {error_msg}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {error_msg}")

    logger.info(f"OCR page successful for {file.filename} page {page}/{result.get('total_pages')}")
    return JSONResponse({"result": result.get("result"), "total_pages": result.get("total_pages")})


def parse_args():
    parser = argparse.ArgumentParser(description="Run PaddleOCR-VL FastAPI service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Number of OCR worker processes")
    parser.add_argument("--device", default="gpu:0", help="Device for Paddle (e.g., gpu:0 or cpu)")
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/.paddlex/official_models"),
        help="Directory containing PaddleOCR-VL models",
    )
    return parser.parse_args()


def main():
    global SERVICE_ARGS
    SERVICE_ARGS = parse_args()
    uvicorn.run(app, host=SERVICE_ARGS.host, port=SERVICE_ARGS.port, log_level="info")


if __name__ == "__main__":
    main()
