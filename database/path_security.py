"""
Path and identifier safety helpers for filesystem operations.
"""

from pathlib import Path
import re


_SAFE_ID_PART = re.compile(r"^[A-Za-z0-9._-]+$")


def normalize_arxiv_id(arxiv_id: str) -> str:
    """
    Validate and normalize arXiv IDs before using them in paths/URLs.

    Supports modern IDs (e.g. 0704.0001) and legacy IDs with one slash
    (e.g. cs.AI/9901001). Rejects traversal/absolute path attempts.
    """
    if not isinstance(arxiv_id, str):
        raise ValueError("arXiv ID must be a string")

    normalized = arxiv_id.strip()
    if not normalized:
        raise ValueError("arXiv ID cannot be empty")
    if "\x00" in normalized:
        raise ValueError("arXiv ID contains null byte")

    normalized = normalized.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("~"):
        raise ValueError("arXiv ID cannot be absolute")

    parts = normalized.split("/")
    if len(parts) > 2:
        raise ValueError("arXiv ID has too many path segments")

    for part in parts:
        if part in {"", ".", ".."}:
            raise ValueError("arXiv ID contains invalid path segment")
        if not _SAFE_ID_PART.fullmatch(part):
            raise ValueError(f"arXiv ID contains unsupported characters: {part!r}")

    return normalized


def safe_pdf_filename(arxiv_id: str) -> str:
    """Create a filesystem-safe PDF filename from an arXiv ID."""
    normalized = normalize_arxiv_id(arxiv_id)
    return f"{normalized.replace('/', '__')}.pdf"


def safe_pdf_path(base_path: Path, arxiv_id: str) -> Path:
    """
    Build a safe PDF path rooted under base_path.
    Raises ValueError if the resolved path would escape base_path.
    """
    resolved_base = Path(base_path).resolve()
    candidate = (resolved_base / safe_pdf_filename(arxiv_id)).resolve()
    if not candidate.is_relative_to(resolved_base):
        raise ValueError("Resolved PDF path escapes base path")
    return candidate

