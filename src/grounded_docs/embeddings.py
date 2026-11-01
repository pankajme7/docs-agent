"""Sentence-Transformers embedding adapter."""
from __future__ import annotations

from functools import cached_property

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 20,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def encode_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        return np.asarray(vector, dtype=np.float32).tolist()
