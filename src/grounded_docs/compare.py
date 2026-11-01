"""Compare saved baseline and optimized evaluation artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _human_score_mean(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("human_score") or "").strip()
            if raw:
                score = float(raw)
                if not 0 <= score <= 2:
                    raise ValueError(f"Human score must be between 0 and 2: {score}")
                values.append(score)
    return round(mean(values), 4) if values else None


def compare_files(baseline_path: Path, optimized_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    optimized = json.loads(optimized_path.read_text(encoding="utf-8"))
    b = baseline["summary"]
    o = optimized["summary"]
    human_before = _human_score_mean(baseline_path.with_suffix(".csv"))
    human_after = _human_score_mean(optimized_path.with_suffix(".csv"))
    if b.get("provider") != o.get("provider") or b.get("model") != o.get("model"):
        raise ValueError("Baseline and optimized runs must use the same provider and model")

    metrics = [
        ("Average prompt tokens", "average_prompt_tokens", "tokens"),
        ("Average completion tokens", "average_completion_tokens", "tokens"),
        ("Estimated total API cost", "total_estimated_cost_usd", "USD"),
        ("Retrieval Hit@k", "retrieval_hit_at_k", "ratio"),
        ("Citation source precision", "citation_source_precision", "ratio"),
        ("Abstention accuracy", "abstention_accuracy", "ratio"),
        ("Lexical fact-coverage proxy", "mean_lexical_fact_coverage_proxy", "ratio"),
    ]
    rows = []
    for label, key, unit in metrics:
        before, after = b.get(key), o.get(key)
        delta = None if before is None or after is None else after - before
        if before not in (None, 0) and after is not None:
            pct = (after - before) / before * 100
        else:
            pct = None
        rows.append({"metric": label, "baseline": before, "optimized": after, "delta": delta, "change_percent": pct, "unit": unit})

    rows.append({
        "metric": "Human review score (0-2)",
        "baseline": human_before,
        "optimized": human_after,
        "delta": None if human_before is None or human_after is None else human_after - human_before,
        "change_percent": None,
        "unit": "score",
    })

    prompt_before, prompt_after = b.get("average_prompt_tokens"), o.get("average_prompt_tokens")
    token_savings_pct = None
    if prompt_before not in (None, 0) and prompt_after is not None:
        token_savings_pct = (prompt_before - prompt_after) / prompt_before * 100

    report = {
        "provider": b.get("provider"),
        "model": b.get("model"),
        "baseline_timestamp": b.get("run_timestamp_utc"),
        "optimized_timestamp": o.get("run_timestamp_utc"),
        "prompt_token_savings_percent": token_savings_pct,
        "metrics": rows,
        "note": "A lower token count is not automatically better if human-reviewed answer quality degrades. Lexical fact coverage is only a rough proxy.",
    }
    markdown = [
        f"# Baseline vs optimized: {b.get('provider')} / {b.get('model')}",
        "",
        "| Metric | Baseline | Optimized | Change |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        before = "—" if row["baseline"] is None else f"{row['baseline']:.4f}" if isinstance(row["baseline"], float) else str(row["baseline"])
        after = "—" if row["optimized"] is None else f"{row['optimized']:.4f}" if isinstance(row["optimized"], float) else str(row["optimized"])
        change = "—" if row["delta"] is None else f"{row['delta']:+.4f}"
        markdown.append(f"| {row['metric']} | {before} | {after} | {change} |")
    markdown.extend(["", f"Prompt-token reduction: {token_savings_pct:.2f}%" if token_savings_pct is not None else "Prompt-token reduction: unavailable", "", report["note"], ""])
    report["markdown"] = "\n".join(markdown)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report["markdown"], encoding="utf-8")
        output_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
