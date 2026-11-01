"""Run a fixed question set and record quality proxies, retrieval, cost, and latency."""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_docs.answer_generator import answer_question
from grounded_docs.config import Settings
from grounded_docs.llm_client import LLMClient
from grounded_docs.retrieval import Retriever
from grounded_docs.metrics import (
    citation_source_precision,
    lexical_fact_coverage,
    retrieval_source_hit,
)

logger = logging.getLogger(__name__)


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not questions:
        raise ValueError(f"No evaluation questions found in {path}")
    return questions


def _mean_or_none(values: list[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return round(mean(valid), 4) if valid else None


def run_evaluation(
    settings: Settings,
    *,
    mode: str = "baseline",
    questions_file: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"baseline", "optimized"}:
        raise ValueError("mode must be 'baseline' or 'optimized'")
    questions_file = questions_file or settings.project_root / "data/evaluations/questions.jsonl"
    output_dir = output_dir or settings.project_root / "data/evaluations/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    top_k = settings.baseline_top_k if mode == "baseline" else settings.optimized_top_k
    context_chars = settings.baseline_context_chars if mode == "baseline" else settings.optimized_context_chars

    rows: list[dict[str, Any]] = []
    # Reuse clients and the loaded embedding model across all questions.
    llm = LLMClient(settings)
    retriever = Retriever(settings)
    for index, item in enumerate(load_questions(questions_file), start=1):
        logger.info("[%d] %s", index, item["id"])
        run = answer_question(
            item["question"], settings, top_k=top_k,
            context_chars=context_chars, mode=mode, llm=llm, retriever=retriever,
        )
        retrieved_urls = list(dict.fromkeys(chunk.source_url for chunk in run.retrieved))
        label_to_url = {f"S{i}": chunk.source_url for i, chunk in enumerate(run.retrieved, start=1)}
        expected_sources = item.get("expected_sources", [])
        row = {
            "id": item["id"],
            "question": item["question"],
            "answerable": item.get("answerable", True),
            "answer": run.answer.answer,
            "source_labels": run.answer.sources,
            "insufficient_context": run.answer.insufficient_context,
            "retrieved_urls": retrieved_urls,
            "expected_fact_lexical_coverage": lexical_fact_coverage(
                run.answer.answer, item.get("expected_facts", [])
            ),
            "retrieval_hit_at_k": retrieval_source_hit(retrieved_urls, expected_sources),
            "citation_source_precision": citation_source_precision(
                run.answer.sources, label_to_url, expected_sources
            ),
            "abstention_correct": run.answer.insufficient_context == (not item.get("answerable", True)),
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "latency_seconds": run.latency_seconds,
            "estimated_cost_usd": run.estimated_cost_usd,
            "provider_usage_available": run.provider_usage_available,
            "human_score": None,  # Fill this after reviewing the saved answers.
        }
        rows.append(row)

    costs = [row["estimated_cost_usd"] for row in rows]
    total_cost = None if not any(cost is not None for cost in costs) else round(sum(cost or 0 for cost in costs), 8)
    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": settings.provider,
        "model": settings.llm_model,
        "mode": mode,
        "top_k": top_k,
        "context_char_limit": context_chars,
        "question_count": len(rows),
        "average_prompt_tokens": _mean_or_none([r["prompt_tokens"] for r in rows]),
        "average_completion_tokens": _mean_or_none([r["completion_tokens"] for r in rows]),
        "average_latency_seconds": _mean_or_none([r["latency_seconds"] for r in rows]),
        "total_estimated_cost_usd": total_cost,
        "retrieval_hit_at_k": _mean_or_none([r["retrieval_hit_at_k"] for r in rows if r["retrieval_hit_at_k"] is not None]),
        "citation_source_precision": _mean_or_none([r["citation_source_precision"] for r in rows if r["citation_source_precision"] is not None]),
        "abstention_accuracy": _mean_or_none([1.0 if r["abstention_correct"] else 0.0 for r in rows]),
        "mean_lexical_fact_coverage_proxy": _mean_or_none([r["expected_fact_lexical_coverage"] for r in rows]),
        "human_score_mean": None,
        "notes": "Lexical fact coverage is a rough proxy, not semantic accuracy. Review answers and fill human_score.",
    }
    safe_model = settings.llm_model.replace("/", "_").replace(":", "_")
    output_path = output_dir / f"{settings.provider}-{safe_model}-{mode}.json"
    output_path.write_text(json.dumps({"summary": summary, "results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                flattened = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
                writer.writerow(flattened)
    return {"summary": summary, "json_path": str(output_path), "csv_path": str(csv_path)}
