"""Token estimation, API cost calculations, and evaluation proxy metrics."""
from __future__ import annotations

import json
import re
from typing import Any



def count_text_tokens(text: str, model: str = "gpt-4.1-mini") -> int:
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Only a fallback for lightweight unit-test environments; install requirements
        # for real measurements. Whitespace token counts are not billing-grade estimates.
        return len(text.split())


def count_message_tokens(messages: list[dict[str, Any]], model: str = "gpt-4.1-mini") -> int:
    # An estimate of serialized message content; provider-reported usage is preferred.
    serialized = json.dumps(messages, ensure_ascii=False, default=str)
    return count_text_tokens(serialized, model)


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_1m: float | None,
    output_price_per_1m: float | None,
) -> float | None:
    if input_price_per_1m is None or output_price_per_1m is None:
        return None
    return (input_tokens * input_price_per_1m + output_tokens * output_price_per_1m) / 1_000_000


def lexical_fact_coverage(answer: str, expected_facts: list[str]) -> float | None:
    """A rough lexical proxy, NOT semantic answer accuracy or a substitute for review."""
    if not expected_facts:
        return None
    answer_terms = set(re.findall(r"[a-z0-9]+", answer.lower()))
    stop_words = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "by", "can", "be", "it", "that", "with", "as", "from"}
    scores = []
    for fact in expected_facts:
        terms = {term for term in re.findall(r"[a-z0-9]+", fact.lower()) if term not in stop_words}
        if terms:
            scores.append(len(terms & answer_terms) / len(terms))
    return sum(scores) / len(scores) if scores else None


def retrieval_source_hit(retrieved_urls: list[str], expected_urls: list[str]) -> bool | None:
    if not expected_urls:
        return None
    return any(url in set(retrieved_urls) for url in expected_urls)


def citation_source_precision(source_labels: list[str], label_to_url: dict[str, str], expected_urls: list[str]) -> float | None:
    if not source_labels:
        return 0.0 if expected_urls else None
    if not expected_urls:
        return None
    cited_urls = {label_to_url[label] for label in source_labels if label in label_to_url}
    return len(cited_urls & set(expected_urls)) / len(cited_urls) if cited_urls else 0.0
