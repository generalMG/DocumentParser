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


def _process_pdf(pdf_bytes: bytes, pages: int, device: str, model_dir: str) -> dict:
    """Worker: load model once per process and run predict. Returns dict to avoid pickling errors."""
    try:
        import paddle
        from paddleocr import PaddleOCRVL  # type: ignore
    except Exception as e:
        return {"ok": False, "error": f"Import failed: {e}"}

    global _MODEL
    try:
        paddle.device.set_device(device)
    except Exception:
        pass

    try:
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

        if isinstance(results, list) and pages and pages > 0:
            results = results[:pages]
        return {"ok": True, "results": _to_serializable(results)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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

    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=f"OCR failed: {result.get('error')}")

    return JSONResponse({"results": result.get("results")})


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
