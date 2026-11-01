"""Convenience entrypoint; equivalent to `grounded-docs benchmark-ollama`."""
from grounded_docs.benchmark import benchmark_models
from grounded_docs.config import Settings

if __name__ == "__main__":
    settings = Settings.from_env()
    output = benchmark_models(settings, [
        "qwen3:4b-instruct-2507-q4_K_M",
        "qwen3:4b-instruct-2507-q8_0",
    ])
    print(output["output_path"])
