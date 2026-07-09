# Worker call buckets

Trace: `trace_20260407_012530_12511.jsonl`

Results: `truthful_qa_benchmark_20260407_012538.jsonl`  
Bench: `truthful_qa`

| Worker calls | Truthful | Untruthful | Total |
|---:|---:|---:|---:|
| 1 | 27 | 26 | 53 |
| 2 | 8 | 7 | 15 |
| 3 | 1 | 2 | 3 |
| 4 | 1 | 7 | 8 |
| 5+ | 2 | 9 | 11 |
| **All** | **39** | **51** | **90** |

## Multi-agent vs single-tool routing

| Routing | Correct | Incorrect | Total |
|---|---:|---:|---:|
| AskMultipleAgents used ≥1 | 1 | 3 | 4 |
| AskAgent only (no multi tool) | 38 | 48 | 86 |
