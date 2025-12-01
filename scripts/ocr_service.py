#!/usr/bin/env python3
"""
FastAPI service that hosts PaddleOCR-VL and processes uploaded PDFs.

- Keeps a pool of PaddleOCR-VL instances (one per process).
- Accepts PDF uploads and returns OCR results as JSON.
- Model paths are read from a local directory (pre-downloaded).
"""

import argparse
import asyncio
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from typing import Any, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv(override=True)

app = FastAPI(title="PaddleOCR-VL Service")

POOL: Optional[ProcessPoolExecutor] = None
SERVICE_ARGS = None


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


def _process_pdf(pdf_bytes: bytes, pages: int, device: str, model_dir: str) -> List[Any]:
    """Worker: load model once per process and run predict."""
    import paddle
    from paddleocr import PaddleOCRVL  # type: ignore

    global _MODEL
    try:
        paddle.device.set_device(device)
    except Exception:
        pass

    if "_MODEL" not in globals() or _MODEL is None:
        _MODEL = PaddleOCRVL(
            use_layout_detection=True,
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            layout_detection_model_dir=os.path.join(model_dir, "PP-DocLayoutV2"),
            vl_rec_model_dir=model_dir,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        results = _MODEL.predict(tmp.name)

    if isinstance(results, list) and pages > 0:
        results = results[:pages]
    return _to_serializable(results)


@app.on_event("startup")
async def _startup():
    global POOL
    if SERVICE_ARGS is None:
        return
    ctx = mp.get_context("spawn")
    POOL = ProcessPoolExecutor(max_workers=SERVICE_ARGS.workers, mp_context=ctx)


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
    pages: int = Form(0),
):
    if POOL is None:
        raise HTTPException(status_code=503, detail="Worker pool not ready")
    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            POOL, _process_pdf, data, pages, SERVICE_ARGS.device, SERVICE_ARGS.model_dir
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    return JSONResponse({"results": result})


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
    uvicorn.run("scripts.ocr_service:app", host=SERVICE_ARGS.host, port=SERVICE_ARGS.port, log_level="info")


if __name__ == "__main__":
    main()
