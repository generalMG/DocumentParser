#!/usr/bin/env python3
"""
Compute Perplexity of OCR Extracted Text
-----------------------------------------
Uses Mistral 7B to measure the perplexity of OCR-extracted text
from arxiv papers stored in the database.

Uses sliding window chunking with overlap so the model can see
previous context when predicting subsequent tokens.

Perplexity measures how well a language model predicts the text.
Lower perplexity = more coherent/natural text.
Higher perplexity = more garbled/noisy text (typical of poor OCR).
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load env BEFORE importing database modules
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from database.database import DatabaseManager, get_database_url_from_env
from database.models import ArxivPaper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Perplexity")

MODEL_NAME = "mistralai/Mistral-7B-v0.1"


def extract_text_from_ocr(ocr_results: Dict[str, Any]) -> str:
    """
    Extract all text content from OCR results JSONB.

    OCR results structure (PaddleOCRVL format):
    {
        "results": [
            {
                "page_index": 0,
                "layout_det_res": {
                    "boxes": [
                        {
                            "label": "text" | "paragraph" | ...,
                            "content": "extracted text...",
                            "score": "0.95",
                            "coordinate": [x1, y1, x2, y2]
                        },
                        ...
                    ]
                }
            },
            ...
        ],
        "total_pages": N
    }
    """
    if not ocr_results:
        return ""

    texts = []
    results = ocr_results.get('results', [])

    for page_result in results:
        if not isinstance(page_result, dict):
            continue

        # Get boxes from layout_det_res
        layout_res = page_result.get('layout_det_res', {})
        boxes = layout_res.get('boxes', [])

        # Sort boxes by vertical position (y1 coordinate) for reading order
        sorted_boxes = sorted(boxes, key=lambda b: b.get('coordinate', [0, 0, 0, 0])[1])

        for box in sorted_boxes:
            if isinstance(box, dict):
                content = box.get('content', '') or box.get('text', '')
                if content and isinstance(content, str):
                    texts.append(content.strip())

    return "\n".join(texts)


def compute_perplexity_sliding_window(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_length: int = 1024,
    stride: int = 512,
    device: str = "cuda",
    penalty_power: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute perplexity using sliding window with overlapping context.

    Returns detailed statistics to avoid smoothing out bad sections.

    Args:
        text: Input text to evaluate
        model: Loaded causal language model
        tokenizer: Model's tokenizer
        context_length: Full window size (model sees this much)
        stride: How many new tokens to score per window
        device: Compute device
        penalty_power: Power for generalized mean (higher = more penalty for bad chunks)

    Returns:
        Dict with perplexity statistics:
        - mean_ppl: Standard average perplexity
        - median_ppl: Median chunk perplexity
        - p90_ppl, p95_ppl, p99_ppl: Percentiles
        - max_ppl: Worst chunk perplexity
        - penalized_ppl: Power mean with penalty_power
        - chunk_ppls: List of per-chunk perplexities
        - num_chunks: Number of chunks processed
    """
    import numpy as np

    empty_result = {
        'mean_ppl': float('inf'),
        'median_ppl': float('inf'),
        'p90_ppl': float('inf'),
        'p95_ppl': float('inf'),
        'p99_ppl': float('inf'),
        'max_ppl': float('inf'),
        'min_ppl': float('inf'),
        'penalized_ppl': float('inf'),
        'chunk_ppls': [],
        'num_chunks': 0,
        'total_tokens': 0,
    }

    if not text.strip():
        return empty_result

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)
    logger.info(f"Total tokens: {seq_len}, context_length: {context_length}, stride: {stride}")

    if seq_len <= 1:
        return empty_result

    # For short sequences, just compute directly
    if seq_len <= context_length:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            ppl = torch.exp(outputs.loss).item()
        return {
            'mean_ppl': ppl,
            'median_ppl': ppl,
            'p90_ppl': ppl,
            'p95_ppl': ppl,
            'p99_ppl': ppl,
            'max_ppl': ppl,
            'min_ppl': ppl,
            'penalized_ppl': ppl,
            'chunk_ppls': [ppl],
            'num_chunks': 1,
            'total_tokens': seq_len,
        }

    # Sliding window for long sequences
    chunk_nlls = []  # NLL per chunk
    chunk_tokens = []  # Tokens scored per chunk
    chunk_ppls = []  # Perplexity per chunk

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + context_length, seq_len)
        chunk_len = end_loc - begin_loc

        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()

        # Determine which tokens to score in this window
        if begin_loc == 0:
            target_chunk[:, 0] = -100
            tokens_to_score = chunk_len - 1
        else:
            context_portion = context_length - stride
            if context_portion > 0 and context_portion < chunk_len:
                target_chunk[:, :context_portion] = -100
                tokens_to_score = chunk_len - context_portion
            else:
                tokens_to_score = chunk_len

        if tokens_to_score <= 0:
            continue

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            chunk_nll = outputs.loss.item()  # Average NLL for this chunk
            chunk_ppl = torch.exp(outputs.loss).item()

        chunk_nlls.append(chunk_nll * tokens_to_score)
        chunk_tokens.append(tokens_to_score)
        chunk_ppls.append(chunk_ppl)

        if end_loc >= seq_len:
            break

    if not chunk_ppls:
        return empty_result

    chunk_ppls_arr = np.array(chunk_ppls)
    total_tokens = sum(chunk_tokens)

    # Standard mean perplexity (from total NLL)
    total_nll = sum(chunk_nlls)
    mean_ppl = np.exp(total_nll / total_tokens)

    # Percentiles and statistics
    median_ppl = np.median(chunk_ppls_arr)
    p90_ppl = np.percentile(chunk_ppls_arr, 90)
    p95_ppl = np.percentile(chunk_ppls_arr, 95)
    p99_ppl = np.percentile(chunk_ppls_arr, 99)
    max_ppl = np.max(chunk_ppls_arr)
    min_ppl = np.min(chunk_ppls_arr)

    # Power mean (generalized mean) - penalizes high values
    # Formula: (mean(x^p))^(1/p) where p > 1 emphasizes larger values
    penalized_ppl = np.power(np.mean(np.power(chunk_ppls_arr, penalty_power)), 1.0 / penalty_power)

    logger.info(f"Processed {len(chunk_ppls)} chunks, {total_tokens} tokens")
    logger.info(f"Mean PPL: {mean_ppl:.2f}, Penalized PPL (p={penalty_power}): {penalized_ppl:.2f}, Max: {max_ppl:.2f}")

    return {
        'mean_ppl': float(mean_ppl),
        'median_ppl': float(median_ppl),
        'p90_ppl': float(p90_ppl),
        'p95_ppl': float(p95_ppl),
        'p99_ppl': float(p99_ppl),
        'max_ppl': float(max_ppl),
        'min_ppl': float(min_ppl),
        'penalized_ppl': float(penalized_ppl),
        'chunk_ppls': chunk_ppls,
        'num_chunks': len(chunk_ppls),
        'total_tokens': total_tokens,
    }


def load_model(model_name: str, device: str = "cuda") -> tuple:
    """Load model and tokenizer from HuggingFace."""
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in float16 for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # Automatic device placement
        trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Model loaded on device(s): {model.hf_device_map if hasattr(model, 'hf_device_map') else device}")

    return model, tokenizer


def get_paper_with_ocr(db: DatabaseManager, paper_id: Optional[str] = None) -> Optional[dict]:
    """Fetch a paper with OCR results from the database."""
    with db.session_scope() as session:
        if paper_id:
            paper = session.query(ArxivPaper).filter(
                ArxivPaper.id == paper_id,
                ArxivPaper.ocr_results.isnot(None)
            ).first()
        else:
            # Get first paper with OCR results
            paper = session.query(ArxivPaper).filter(
                ArxivPaper.ocr_processed == True,
                ArxivPaper.ocr_results.isnot(None)
            ).first()

        if not paper:
            return None

        return {
            'id': paper.id,
            'title': paper.title,
            'ocr_results': paper.ocr_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Compute perplexity of OCR text using Mistral 7B")
    parser.add_argument('--paper-id', type=str, help="Specific paper ID to analyze")
    parser.add_argument('--model', type=str, default=MODEL_NAME, help="HuggingFace model name")
    parser.add_argument('--context-length', type=int, default=1024,
                        help="Context window size (tokens model sees at once)")
    parser.add_argument('--stride', type=int, default=512,
                        help="New tokens to score per window (overlap = context_length - stride)")
    parser.add_argument('--penalty-power', type=float, default=2.0,
                        help="Power for penalized mean (higher = more penalty for bad chunks)")
    parser.add_argument('--show-worst', type=int, default=5,
                        help="Show N worst chunks")
    parser.add_argument('--device', type=str, default="cuda", help="Compute device")
    args = parser.parse_args()

    # Initialize database
    db = DatabaseManager(db_url=get_database_url_from_env())

    # Fetch paper
    logger.info("Fetching paper from database...")
    paper = get_paper_with_ocr(db, args.paper_id)

    if not paper:
        logger.error("No paper with OCR results found")
        sys.exit(1)

    logger.info(f"Paper: {paper['id']} - {paper['title'][:60]}...")

    # Extract text
    text = extract_text_from_ocr(paper['ocr_results'])
    word_count = len(text.split())
    char_count = len(text)

    logger.info(f"Extracted text: {word_count} words, {char_count} characters")

    if not text.strip():
        logger.error("No text extracted from OCR results")
        sys.exit(1)

    # Preview
    preview = text[:500].replace('\n', ' ')
    logger.info(f"Text preview: {preview}...")

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Compute perplexity
    logger.info("Computing perplexity with sliding window...")
    overlap = args.context_length - args.stride
    logger.info(f"Context: {args.context_length} tokens, Stride: {args.stride}, Overlap: {overlap} tokens")

    results = compute_perplexity_sliding_window(
        text,
        model,
        tokenizer,
        context_length=args.context_length,
        stride=args.stride,
        device=args.device,
        penalty_power=args.penalty_power,
    )

    # Report
    print("\n" + "="*70)
    print("PERPLEXITY RESULTS")
    print("="*70)
    print(f"Paper ID:       {paper['id']}")
    print(f"Title:          {paper['title'][:52]}...")
    print(f"Word count:     {word_count}")
    print(f"Tokens:         {results['total_tokens']}")
    print(f"Chunks:         {results['num_chunks']}")
    print(f"Model:          {args.model}")
    print(f"Context/Stride: {args.context_length}/{args.stride} (overlap: {overlap})")
    print("-"*70)
    print("PERPLEXITY STATISTICS:")
    print(f"  Mean PPL:       {results['mean_ppl']:>10.2f}  (standard average)")
    print(f"  Median PPL:     {results['median_ppl']:>10.2f}  (middle chunk)")
    print(f"  Min PPL:        {results['min_ppl']:>10.2f}  (best chunk)")
    print(f"  Max PPL:        {results['max_ppl']:>10.2f}  (worst chunk)")
    print("-"*70)
    print("PERCENTILES (tail behavior):")
    print(f"  90th:           {results['p90_ppl']:>10.2f}")
    print(f"  95th:           {results['p95_ppl']:>10.2f}")
    print(f"  99th:           {results['p99_ppl']:>10.2f}")
    print("-"*70)
    print(f"PENALIZED PPL:    {results['penalized_ppl']:>10.2f}  (power={args.penalty_power})")
    print("="*70)

    # Show worst chunks
    if args.show_worst > 0 and results['chunk_ppls']:
        print(f"\nWORST {args.show_worst} CHUNKS:")
        sorted_chunks = sorted(enumerate(results['chunk_ppls']), key=lambda x: x[1], reverse=True)
        for i, (chunk_idx, ppl) in enumerate(sorted_chunks[:args.show_worst]):
            token_start = chunk_idx * args.stride
            token_end = min(token_start + args.context_length, results['total_tokens'])
            print(f"  {i+1}. Chunk {chunk_idx:3d} (tokens {token_start:5d}-{token_end:5d}): PPL = {ppl:.2f}")

    # Interpretation guide
    print("\nINTERPRETATION (using penalized PPL):")
    ppl = results['penalized_ppl']
    if ppl < 10:
        print("  Very low  - Extremely coherent text")
    elif ppl < 50:
        print("  Low       - High quality, coherent text")
    elif ppl < 100:
        print("  Moderate  - Decent text quality with some noise")
    elif ppl < 500:
        print("  High      - Noticeable OCR errors or non-standard text")
    else:
        print("  Very high - Significant OCR errors or garbled text")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
