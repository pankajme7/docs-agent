"""Run the same evaluation against multiple Ollama quantizations and sample memory."""
from __future__ import annotations

import json
import platform
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import psutil

from grounded_docs.config import Settings
from grounded_docs.evaluation import run_evaluation


def _find_ollama_daemon() -> psutil.Process | None:
    for process in psutil.process_iter(["name"]):
        try:
            name = (process.info.get("name") or "").lower()
            if "ollama" in name:
                return process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class ProcessTreeMemorySampler:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval = interval_seconds
        self.peak_bytes = 0
        self.process_found = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            daemon = _find_ollama_daemon()
            if daemon:
                self.process_found = True
                processes = [daemon]
                try:
                    processes.extend(daemon.children(recursive=True))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                total = 0
                for process in processes:
                    try:
                        total += process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                self.peak_bytes = max(self.peak_bytes, total)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def benchmark_models(
    settings: Settings,
    models: list[str],
    mode: str = "optimized",
) -> dict[str, Any]:
    if settings.provider != "ollama":
        raise ValueError("Set LLM_PROVIDER=ollama before running the local benchmark.")
    results = []
    for model in models:
        model_settings = replace(settings, provider="ollama", ollama_model=model)
        sampler = ProcessTreeMemorySampler()
        sampler.start()
        started = time.perf_counter()
        try:
            evaluation = run_evaluation(model_settings, mode=mode)
        finally:
            sampler.stop()
        elapsed = time.perf_counter() - started
        summary = evaluation["summary"]
        evaluation_rows = json.loads(Path(evaluation["json_path"]).read_text(encoding="utf-8"))["results"]
        total_output_tokens = sum(row["completion_tokens"] for row in evaluation_rows)
        warm_latencies = [row["latency_seconds"] for row in evaluation_rows[1:]]
        results.append({
            "model": model,
            "mode": mode,
            "evaluation_seconds": round(elapsed, 3),
            "peak_ollama_process_tree_rss_gb": round(sampler.peak_bytes / (1024 ** 3), 3) if sampler.process_found else None,
            "process_memory_measurement_available": sampler.process_found,
            "average_latency_seconds": summary["average_latency_seconds"],
            "first_question_latency_seconds": evaluation_rows[0]["latency_seconds"] if evaluation_rows else None,
            "warm_average_latency_seconds": round(sum(warm_latencies) / len(warm_latencies), 4) if warm_latencies else None,
            "total_output_tokens": total_output_tokens,
            "aggregate_output_tokens_per_second": round(total_output_tokens / elapsed, 3) if elapsed else None,
            "retrieval_hit_at_k": summary["retrieval_hit_at_k"],
            "citation_source_precision": summary["citation_source_precision"],
            "abstention_accuracy": summary["abstention_accuracy"],
            "mean_lexical_fact_coverage_proxy": summary["mean_lexical_fact_coverage_proxy"],
        })
    payload = {
        "hardware": {
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True),
            "total_system_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        },
        "results": results,
        "notes": "RSS is the sampled Ollama daemon plus its child processes when discoverable. Aggregate output tokens/sec includes retrieval and the full evaluation, not pure decode throughput.",
    }
    output = settings.project_root / "data/evaluations/results/ollama-benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"payload": payload, "output_path": str(output)}
