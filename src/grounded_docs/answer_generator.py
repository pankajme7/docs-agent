"""Grounded RAG answer generation with source labels and measurement metadata."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from grounded_docs.config import Settings
from grounded_docs.llm_client import LLMClient, parse_grounded_answer
from grounded_docs.metrics import count_message_tokens, calculate_cost
from grounded_docs.prompts import JSON_INSTRUCTION, SYSTEM_PROMPT, STREAMING_PROMPT
from grounded_docs.retrieval import Retriever
from grounded_docs.schemas import GroundedAnswer, RetrievedChunk


@dataclass
class AnswerRun:
    question: str
    answer: GroundedAnswer
    retrieved: list[RetrievedChunk]
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    provider_usage_available: bool
    estimated_cost_usd: float | None
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer.model_dump(),
            "retrieved": [item.model_dump() for item in self.retrieved],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_seconds": round(self.latency_seconds, 4),
            "provider_usage_available": self.provider_usage_available,
            "estimated_cost_usd": self.estimated_cost_usd,
            "mode": self.mode,
        }


def build_context(chunks: list[RetrievedChunk], max_chars: int) -> tuple[str, dict[str, str]]:
    sections: list[str] = []
    label_to_url: dict[str, str] = {}
    remaining = max_chars
    for index, chunk in enumerate(chunks, start=1):
        label = f"S{index}"
        label_to_url[label] = chunk.source_url
        header = f"[{label}] Title: {chunk.title}\nSource: {chunk.source_url}\n"
        body = chunk.text.strip()
        block = f"{header}{body}"
        if len(block) > remaining:
            if remaining <= len(header) + 100:
                break
            block = header + body[: remaining - len(header)]
        sections.append(block)
        remaining -= len(block) + 2
        if remaining <= 0:
            break
    return "\n\n---\n\n".join(sections), label_to_url


def _run_cost(settings: Settings, prompt_tokens: int, completion_tokens: int) -> float | None:
    if settings.provider == "ollama":
        return 0.0  # API bill is zero; electricity and hardware depreciation are excluded.
    return calculate_cost(
        prompt_tokens,
        completion_tokens,
        settings.openai_input_price_per_1m,
        settings.openai_output_price_per_1m,
    )


def answer_question(
    question: str,
    settings: Settings,
    *,
    top_k: int | None = None,
    context_chars: int | None = None,
    mode: str = "default",
    llm: LLMClient | None = None,
    retriever: Retriever | None = None,
) -> AnswerRun:
    llm = llm or LLMClient(settings)
    retriever = retriever or Retriever(settings)
    started = time.perf_counter()
    retrieved = retriever.search(question, top_k or settings.default_top_k)
    context, label_to_url = build_context(retrieved, context_chars or settings.optimized_context_chars)
    if not context:
        empty = GroundedAnswer(
            answer="The documentation index is empty. Run `grounded-docs ingest` first.",
            sources=[], insufficient_context=True,
        )
        return AnswerRun(question, empty, retrieved, 0, 0, time.perf_counter() - started, False, None, mode)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + JSON_INSTRUCTION},
        {"role": "user", "content": f"Documentation context:\n\n{context}\n\nQuestion: {question}"},
    ]
    result = llm.complete(messages, json_mode=True)
    parsed = parse_grounded_answer(result.content)
    # Ignore source labels the model invented that were not present in this run.
    parsed.sources = [source for source in parsed.sources if source in label_to_url]
    # Keep a total duration including retrieval and context preparation.
    total_latency = time.perf_counter() - started
    return AnswerRun(
        question=question,
        answer=parsed,
        retrieved=retrieved,
        prompt_tokens=result.prompt_tokens or count_message_tokens(messages, settings.llm_model),
        completion_tokens=result.completion_tokens,
        latency_seconds=total_latency,
        provider_usage_available=result.provider_usage_available,
        estimated_cost_usd=_run_cost(settings, result.prompt_tokens, result.completion_tokens),
        mode=mode,
    )


def stream_answer(question: str, settings: Settings) -> None:
    llm = LLMClient(settings)
    retriever = Retriever(settings)
    chunks = retriever.search(question, settings.default_top_k)
    context, label_to_url = build_context(chunks, settings.optimized_context_chars)
    if not context:
        print("The documentation index is empty. Run `grounded-docs ingest` first.")
        return
    messages = [
        {"role": "system", "content": STREAMING_PROMPT},
        {"role": "user", "content": f"Documentation context:\n\n{context}\n\nQuestion: {question}"},
    ]
    for token in llm.stream(messages):
        print(token, end="", flush=True)
    print("\n\nSources")
    for label, url in label_to_url.items():
        print(f"[{label}] {url}")
