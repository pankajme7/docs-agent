"""Central configuration loaded from environment variables and an optional .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _optional_float(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    project_root: Path
    provider: str
    openai_api_key: str
    openai_model: str
    ollama_base_url: str
    ollama_api_key: str
    ollama_model: str
    embedding_model: str
    chroma_path: Path
    chroma_collection: str
    default_top_k: int
    baseline_top_k: int
    optimized_top_k: int
    baseline_context_chars: int
    optimized_context_chars: int
    chunk_words: int
    chunk_overlap_words: int
    mcp_host: str
    mcp_port: int
    mcp_url: str
    max_output_tokens: int
    temperature: float
    openai_input_price_per_1m: float | None
    openai_output_price_per_1m: float | None

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        load_dotenv(root / ".env", override=False)
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        if provider not in {"openai", "ollama"}:
            raise ValueError("LLM_PROVIDER must be either 'openai' or 'ollama'.")
        return cls(
            project_root=root,
            provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            ollama_api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            chroma_path=root / os.getenv("CHROMA_PATH", "data/chroma"),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "kubernetes_docs"),
            default_top_k=_int("DEFAULT_TOP_K", 4),
            baseline_top_k=_int("BASELINE_TOP_K", 8),
            optimized_top_k=_int("OPTIMIZED_TOP_K", 4),
            baseline_context_chars=_int("BASELINE_CONTEXT_CHARS", 14000),
            optimized_context_chars=_int("OPTIMIZED_CONTEXT_CHARS", 6500),
            chunk_words=_int("CHUNK_WORDS", 145),
            chunk_overlap_words=_int("CHUNK_OVERLAP_WORDS", 28),
            mcp_host=os.getenv("MCP_HOST", "127.0.0.1"),
            mcp_port=_int("MCP_PORT", 8000),
            mcp_url=os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp"),
            max_output_tokens=_int("MAX_OUTPUT_TOKENS", 900),
            temperature=_float("TEMPERATURE", 0.1),
            openai_input_price_per_1m=_optional_float("OPENAI_INPUT_PRICE_PER_1M"),
            openai_output_price_per_1m=_optional_float("OPENAI_OUTPUT_PRICE_PER_1M"),
        )

    @property
    def llm_model(self) -> str:
        return self.openai_model if self.provider == "openai" else self.ollama_model

    @property
    def llm_base_url(self) -> str | None:
        return None if self.provider == "openai" else self.ollama_base_url

    @property
    def llm_api_key(self) -> str:
        return self.openai_api_key if self.provider == "openai" else self.ollama_api_key
