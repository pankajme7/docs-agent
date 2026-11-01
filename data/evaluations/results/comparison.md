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
