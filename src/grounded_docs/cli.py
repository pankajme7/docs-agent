"""Command-line interface for ingestion, questions, evaluation, and local benchmarking."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from grounded_docs.answer_generator import answer_question, stream_answer
from grounded_docs.benchmark import benchmark_models
from grounded_docs.config import Settings
from grounded_docs.compare import compare_files
from grounded_docs.evaluation import run_evaluation
from grounded_docs.ingest import ingest
from grounded_docs.tool_agent import ask_with_mcp


def _print_answer(run) -> None:
    print("\nAnswer\n------")
    print(run.answer.answer)
    if run.answer.sources:
        print("\nSources")
        for label in run.answer.sources:
            index = int(label[1:]) - 1
            if 0 <= index < len(run.retrieved):
                print(f"[{label}] {run.retrieved[index].source_url}")
    print("\nRun metrics")
    print(json.dumps({
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "latency_seconds": round(run.latency_seconds, 3),
        "provider_usage_available": run.provider_usage_available,
        "estimated_cost_usd": run.estimated_cost_usd,
        "insufficient_context": run.answer.insufficient_context,
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grounded-docs", description="Evaluated RAG + MCP documentation agent")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Download documentation and build the Chroma index")
    ingest_parser.add_argument("--reset", action="store_true", help="Delete and rebuild the current collection")
    ingest_parser.add_argument("--sources", type=Path, default=None, help="Optional alternate sources.yaml")

    ask_parser = sub.add_parser("ask", help="Ask a question using direct RAG or MCP tool calling")
    ask_parser.add_argument("question", help="Question to answer")
    ask_parser.add_argument("--stream", action="store_true", help="Stream a plain-text answer")
    ask_parser.add_argument("--json", action="store_true", help="Return a validated structured JSON answer")
    ask_parser.add_argument("--agent", action="store_true", help="Use LLM tool calling through the MCP server")
    ask_parser.add_argument("--top-k", type=int, default=None)

    eval_parser = sub.add_parser("evaluate", help="Run the fixed evaluation set")
    eval_parser.add_argument("--mode", choices=["baseline", "optimized"], default="baseline")
    eval_parser.add_argument("--questions", type=Path, default=None)

    sub.add_parser("mcp-server", help="Start the MCP Streamable HTTP server")

    compare_parser = sub.add_parser("compare", help="Compare saved baseline and optimized evaluation runs")
    compare_parser.add_argument("baseline", type=Path, help="Baseline evaluation JSON path")
    compare_parser.add_argument("optimized", type=Path, help="Optimized evaluation JSON path")
    compare_parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path")

    bench_parser = sub.add_parser("benchmark-ollama", help="Compare two or more local Ollama models")
    bench_parser.add_argument(
        "--models", nargs="+",
        default=["qwen3:4b-instruct-2507-q4_K_M", "qwen3:4b-instruct-2507-q8_0"],
    )
    bench_parser.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()

    try:
        if args.command == "ingest":
            result = ingest(settings, args.sources, reset=args.reset)
            print(json.dumps(result, indent=2))
        elif args.command == "ask":
            if args.agent:
                print(ask_with_mcp(args.question, settings))
            elif args.stream:
                stream_answer(args.question, settings)
            else:
                run = answer_question(args.question, settings, top_k=args.top_k)
                if args.json:
                    print(json.dumps(run.answer.model_dump(), indent=2, ensure_ascii=False))
                else:
                    _print_answer(run)
        elif args.command == "evaluate":
            result = run_evaluation(settings, mode=args.mode, questions_file=args.questions)
            print(json.dumps(result, indent=2))
        elif args.command == "compare":
            result = compare_files(args.baseline, args.optimized, args.output)
            print(result["markdown"])
        elif args.command == "mcp-server":
            from grounded_docs.mcp_server import mcp
            mcp.run(transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port)
        elif args.command == "benchmark-ollama":
            result = benchmark_models(settings, args.models, mode=args.mode)
            print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        logging.exception("Command failed")
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
