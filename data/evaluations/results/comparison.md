# Baseline vs optimized: ollama / qwen3:4b-instruct-2507-q4_K_M

| Metric | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Average prompt tokens | 1782.9500 | 968.7000 | -814.2500 |
| Average completion tokens | 153.4500 | 134.3500 | -19.1000 |
| Estimated total API cost | 0 | 0 | +0.0000 |
| Retrieval Hit@k | 1.0000 | 0.9444 | -0.0556 |
| Citation source precision | 0.8981 | 0.8611 | -0.0370 |
| Abstention accuracy | 1.0000 | 0.9000 | -0.1000 |
| Lexical fact-coverage proxy | 0.6571 | 0.6278 | -0.0293 |
| Human review score (0-2) | 2.0000 | 1.7500 | -0.2500 |

Prompt-token reduction: 45.67%

A lower token count is not automatically better if human-reviewed answer quality degrades. Lexical fact coverage is only a rough proxy.

## Ollama benchmark results

Hardware: Windows-10-10.0.26300-SP0, 8 CPUs, 15.55 GB system RAM.

| Metric | qwen3:4b-instruct-2507-q4_K_M (optimized) | qwen3:4b-instruct-2507-q8_0 (optimized) |
|---|---:|---:|
| Evaluation time (seconds) | 606.404 | 1342.137 |
| Peak Ollama process-tree RSS (GB) | 5.098 | 6.246 |
| Average latency (seconds) | 30.1386 | 67.0523 |
| First-question latency (seconds) | 18.7140 | 58.5993 |
| Warm average latency (seconds) | 30.7399 | 67.4972 |
| Total output tokens | 2682 | 2709 |
| Aggregate output tokens/second | 4.423 | 2.018 |
| Retrieval Hit@k | 0.9444 | 0.9444 |
| Citation source precision | 0.8889 | 0.9167 |
| Abstention accuracy | 0.9000 | 0.9500 |
| Mean lexical fact-coverage proxy | 0.6540 | 0.6400 |

RSS is the sampled Ollama daemon plus its child processes when discoverable. Aggregate output tokens/second includes retrieval and the full evaluation, not pure decode throughput.
