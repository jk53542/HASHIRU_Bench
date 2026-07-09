# ToolBench × HASHIRU (benchmarking_toolbench.py)

This runner feeds [ToolBench](https://github.com/OpenBMB/ToolBench) **instruction JSON** (`query`, `query_id`, `api_list`) into **HASHIRU** over Gradio and writes results under `HASHIRU_Bench/results/toolbench/`, in the same spirit as BFCL, τ²-bench, and IFBench.

## Why two (or three) environments?

| Component | Venv | Purpose |
|-----------|------|--------|
| **HASHIRU** | HASHIRU `.venv` | `python app.py --no-auth` — Gradio API |
| **This script** | Minimal | `pip install gradio_client python-dotenv` (and `benchmarking_ifbench` imports — same folder) |
| **ToolBench** | ToolBench `.venv` | Official training / `qa_pipeline` / **ToolEval** only if you want paper-style scores |

ToolBench’s `requirements.txt` conflicts with HASHIRU (Gradio 3 vs 5, Pydantic 1 vs 2, etc.). **Do not** install ToolBench into HASHIRU’s venv.

## Prerequisites

1. **Instruction source** — either:
   - **Hugging Face (default):** `tuandunghcmut/toolbench-v1` benchmark splits pulled via `datasets` at runtime:
     ```bash
     pip install datasets
     python benchmarking_toolbench.py --dataset-source hf
     ```
   - You can select split(s):
     ```bash
     python benchmarking_toolbench.py --dataset-source hf \
       --hf-splits g1_instruction g1_category
     ```
   - **Official data:** clone [ToolBench](https://github.com/OpenBMB/ToolBench) and download **`data.zip`** so you have e.g. `data/test_instruction/G1_instruction.json`, **or**
   - **Bundled samples (no large download):** this repo includes  
     `bench/ToolBench/data/test_instruction/sample_toolbench_instructions.json`  
     (synthetic rows matching the same `query` / `query_id` / `api_list` shape).  
     See `data/test_instruction/README.md`. Run with:
     ```bash
     python benchmarking_toolbench.py --toolbench-root ./ToolBench \
       --instruction-files ./ToolBench/data/test_instruction/sample_toolbench_instructions.json
     ```

2. Start HASHIRU (from its own venv):

   ```bash
   cd /path/to/HASHIRU_modified
   source .venv/bin/activate
   python app.py --no-auth
   ```

3. Install only what the **bench** runner needs (from any clean venv or global):

   ```bash
   pip install gradio_client python-dotenv
   ```

## Run

From `HASHIRU_Bench/bench/`:

```bash
# Default (Hugging Face benchmark dataset)
python benchmarking_toolbench.py --dataset-source hf

# Local JSON mode
python benchmarking_toolbench.py --dataset-source local --toolbench-root ./ToolBench

# Explicit files
python benchmarking_toolbench.py \
  --toolbench-root ~/ToolBench \
  --instruction-files ~/ToolBench/data/test_instruction/G1_instruction.json

# Smoke test (first 3 queries)
python benchmarking_toolbench.py --toolbench-root ./ToolBench --max-queries 3

# Minimal HASHIRU modes (empty modeIndexes)
python benchmarking_toolbench.py --toolbench-root ./ToolBench --empty-modes
```

Environment variable: `HASHIRU_GRADIO_URL` (default `http://127.0.0.1:7860`).

## Output layout

```
HASHIRU_Bench/results/toolbench/toolbench_<YYYYMMDD_HHMMSS>/
  meta.json                      # URLs, files, flags
  result/
    toolbench_hashiru.jsonl      # one JSON per line (query, response, timing, errors)
    summary.json                 # counts and per-file stats
```

Each JSONL line includes: `source_file`, `query_id`, `query`, `api_list`, `hashiru_response`, `elapsed_sec`, `error`.

## Official ToolEval (optional, ToolBench venv)

This script records **chat final text**, not ToolBench’s internal `qa_pipeline` trajectory files (`*_DFS@1.json` trees). To compute **Pass rate / Win rate** from the paper you must either:

- Run HASHIRU through a custom adapter that matches ToolBench’s prediction format, then use `toolbench/tooleval/` scripts, **or**
- Use ToolBench’s harness with another backbone and treat HASHIRU results here as a **separate** qualitative / ablation track.

If you already have ToolBench installed in a second venv, activate it and follow [ToolBench — ToolEval](https://github.com/OpenBMB/ToolBench#tooleval) after converting predictions (see `convert_to_answer_format.py` in their repo).

Record your ToolBench interpreter for documentation:

```bash
python benchmarking_toolbench.py --toolbench-root ./ToolBench \
  --toolbench-python /path/to/ToolBench/.venv/bin/python
```

That path is stored in `meta.json` only as a hint; this runner does **not** subprocess ToolEval.

## Differences vs native ToolBench

- **No RapidAPI / StableToolBench simulator** in this script: HASHIRU uses its **own** tools (search, files, agents). Scores are **not** comparable to official ToolBench leaderboard numbers without extra alignment work.
- Use this for **repeatable HASHIRU runs** on the same **instruction text** as ToolBench and for **side-by-side** analysis with other HASHIRU benchmarks (BFCL, IFBench, τ²).
