# BFCL runs with HASHIRU — reading results

## Why `multi_turn_long_context` (or other `multi_turn_*`) can show **0% accuracy**

Example run folders: `results/bfcl/bfcl_20260322_142748/`, `results/bfcl/bfcl_20260322_143624/`.

1. **Turn-count mismatch (`multi_turn:force_terminated`)**  
   The evaluator expects the model’s **number of outer user/assistant turns** to match the gold scenario (often 2–5).  
   BFCL’s harness allows only a **limited number of inner steps per user message** (~20). If the model keeps emitting invalid or redundant calls (e.g. wrong API shape, `mv` before `cd`, repeating the same error), it may **never finish the first user turn**. The recorded result then has **one** outer “turn” while gold has **N** → automatic failure, not necessarily a bug in HASHIRU core.

2. **Observed failure modes (e.g. `bfcl_20260322_143624`)**  
   - **Decode is OK** — inference logs show `Successfully decoded model response`; this is not an `ast_parse` / formatting bug in the usual sense.  
   - **Repeating after success** — e.g. `mv` succeeds once (`'final_report.pdf' moved to 'temp'`) then the model issues the same `mv` again until the step cap.  
   - **Long-context directory noise** — `multi_turn_long_context` injects many filler filenames into `ls()` output (BFCL design). Models that loop `ls()` forever never advance.  
   - **Wrong call order** — gold often does `cd(folder='document')` before `mkdir`/`mv`; doing `mkdir` at workspace root then `mv` without `cd` first yields “No such file”.

3. **Wrong call shape / order**  
   Gold traces use specific function names and argument names from the Gorilla schema (e.g. `find(path='.', name='test')`). Calling `mkdir` before navigating into the folder the user named will fail simulation checks.

4. **Benchmark-only mitigations**  
   `benchmarking_bfcl.py` adds extra prompt text (allowed API names, simulator rules, long-context listing hints, “stop repeating after success”) **only in the benchmark client** so you can compare against **unmodified** HASHIRU `manager.py`. Optional: `--bfcl-max-inner-steps N` patches BFCL’s inner step limit (unofficial vs leaderboard defaults).

## Are worker agents supposed to run?

For **`multi_turn_*`** and **`agentic_*`** categories, the benchmark uses **BFCL-direct** mode: Gradio `modeIndexes` is **empty**, so HASHIRU shows **“Selected modes: []”** and **no CEO/worker agents**. That is **correct**: the official BFCL harness needs a **plain-text Python function call** every inner step, not agent traces. Low scores here reflect the **base model’s** FC behavior, not “agents failing.”

To exercise agents against BFCL-style tasks, use a **single-turn** category, e.g.  
`python benchmarking_bfcl.py --test-category simple_python --ceo-force-agents`.

## Clearer console output

- bfcl_eval often prints only `Turn: … Step: …` and Gradio “Loaded as API”. Use **`--verbose-bfcl`** to print each step’s extracted FC line and approximate prompt size to **stderr**.
- **`--summarize-long-bfcl-tool-json`** trims huge `tool` JSON in the **HASHIRU prompt only** (long `ls` listings, long `grep`/`cat` text). The BFCL simulator still runs on full tool output.

## Comparing to “original” HASHIRU

Keep BFCL-specific behavior in `HASHIRU_Bench/bench/` (this script, handler, env). Avoid patching `HASHIRU_modified/src/manager/manager.py` for BFCL so numeric scores stay comparable to the upstream project.

## Useful artifacts

- `result/.../BFCL_v4_*_result.json` — raw model outputs per category  
- `score/.../BFCL_v4_*_score.json` — official BFCL metrics and failure strings  
