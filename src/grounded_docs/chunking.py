"""Paragraph-aware chunking with a configurable word overlap."""
from __future__ import annotations

import re


def chunk_text(text: str, chunk_words: int = 145, overlap_words: int = 28) -> list[str]:
    """Split text into fixed word windows with overlap.

    Word counts are a lightweight proxy for model tokens. Keeping chunks modest is
    important because embedding models have a finite input context window.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")

    words = text.split()
    if not words:
        return []
    step = chunk_words - overlap_words
    chunks = [
        " ".join(words[start : start + chunk_words])
        for start in range(0, len(words), step)
    ]
    return [chunk for chunk in chunks if chunk]
