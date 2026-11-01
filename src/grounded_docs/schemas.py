"""Shared application schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Answer supported by the retrieved documentation.")
    sources: list[str] = Field(
        description="Source IDs such as S1 and S2 that support the answer.",
    )
    insufficient_context: bool = Field(
        description="True when the supplied documentation does not support an answer."
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    title: str
    source_url: str
    section: str = ""
    distance: float | None = None
