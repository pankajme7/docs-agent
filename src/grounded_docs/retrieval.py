"""Retrieval service used by both direct RAG and the MCP tool."""
from __future__ import annotations

import logging

from grounded_docs.config import Settings
from grounded_docs.embeddings import EmbeddingModel
from grounded_docs.logging_utils import log_payload
from grounded_docs.schemas import RetrievedChunk
from grounded_docs.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_model = EmbeddingModel(settings.embedding_model)
        self.store = VectorStore(settings)

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        log_payload(logger, "RAG retrieval input", {"query": query, "top_k": top_k})
        vector = self.embedding_model.encode_query(query)
        raw_results = self.store.query(vector, top_k)
        chunks = [
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                title=str(item["metadata"].get("title", "Untitled")),
                source_url=str(item["metadata"].get("source_url", "")),
                section=str(item["metadata"].get("section", "")),
                distance=item["distance"],
            )
            for item in raw_results
        ]
        log_payload(
            logger,
            "RAG retrieval output",
            {"count": len(chunks), "chunks": [chunk.model_dump() for chunk in chunks]},
        )
        return chunks
