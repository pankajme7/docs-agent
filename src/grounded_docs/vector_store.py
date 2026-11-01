"""Thin ChromaDB adapter; embedding generation remains explicit in this project."""
from __future__ import annotations

from typing import Any

import chromadb

from grounded_docs.config import Settings


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection_name = settings.chroma_collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace_source(
        self,
        source_url: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        self.collection.delete(where={"source_url": source_url})
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    def query(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        total = self.collection.count()
        if total == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(1, top_k), total),
            include=["documents", "metadatas", "distances"],
        )
        rows: list[dict[str, Any]] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for item_id, document, metadata, distance in zip(ids, docs, metas, distances):
            rows.append({
                "chunk_id": item_id,
                "text": document or "",
                "metadata": metadata or {},
                "distance": float(distance) if distance is not None else None,
            })
        return rows

    def count(self) -> int:
        return self.collection.count()
