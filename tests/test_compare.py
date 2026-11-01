import json
from pathlib import Path

from grounded_docs.compare import compare_files


def test_compare_reports_prompt_token_savings(tmp_path: Path):
    base = {"summary": {"provider": "openai", "model": "m", "average_prompt_tokens": 100, "average_completion_tokens": 20, "total_estimated_cost_usd": 1.0, "retrieval_hit_at_k": 0.9, "citation_source_precision": 0.8, "abstention_accuracy": 1.0, "mean_lexical_fact_coverage_proxy": 0.7}}
    opt = {"summary": {"provider": "openai", "model": "m", "average_prompt_tokens": 70, "average_completion_tokens": 18, "total_estimated_cost_usd": 0.6, "retrieval_hit_at_k": 0.9, "citation_source_precision": 0.8, "abstention_accuracy": 1.0, "mean_lexical_fact_coverage_proxy": 0.7}}
    base_path, opt_path = tmp_path / "base.json", tmp_path / "opt.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    opt_path.write_text(json.dumps(opt), encoding="utf-8")
    result = compare_files(base_path, opt_path)
    assert result["prompt_token_savings_percent"] == 30.0
    assert "Prompt-token reduction: 30.00%" in result["markdown"]
    assert result["markdown"].count("\n") >= 5
