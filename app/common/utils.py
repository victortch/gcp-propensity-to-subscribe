"""Miscellaneous utilities shared by training and inference code.

Highlights:
- :func:`get_git_sha` - best-effort current Git commit SHA (``None`` if unavailable).
- :func:`file_sha_or_none` - safe SHA-256 helper that tolerates missing files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def get_logger(name: str, level: str | int = "INFO") -> logging.Logger:
    """
    Create a simple stdout logger. Harmless for training behavior.
    """
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------
# Run / time helpers (metadata only)
# ---------------------------------------------------------------------

def make_run_id() -> str:
    """
    Time-based run ID for organizing artifacts, e.g. 20251104T174530Z.
    """
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def utcnow_iso() -> str:
    """UTC timestamp in ISO format (for metadata fields)."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ---------------------------------------------------------------------
# Hash / SHA helpers (governance only)
# ---------------------------------------------------------------------

def sha256_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def dict_to_sorted_json(d: Dict[str, Any]) -> str:
    """
    Deterministic JSON string (keys sorted). Handy before hashing.
    """
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_sha_or_none(path: str | Path) -> Optional[str]:
    """
    Return SHA-256 of file, or None if file missing.
    """
    p = Path(path)
    if not p.exists():
        return None
    return sha256_file(p)


def get_git_sha() -> Optional[str]:
    """
    Best-effort current Git commit SHA. Returns None if not in a git repo
    or git is unavailable (typical inside Vertex AI containers).
    """
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None
