# StrategyQA: ChatGPT vs small workers (token + accuracy)

**Dataset merge key.** Each shard repeats local `question_num` 1…100. Rows are stitched using trace `question_id` from the winning `ceo_final_answer` (`bench_attempt`, then `ts`). Canonical ints **0…299** = first 300 StrategyQA IDs in this sweep.

## Discovery (all files, mtime desc)

### StrategyQA benchmark JSONLs

| File | Size (bytes) | mtime (local) |
|------|-------------|---------------|
| `strategyqa_benchmark_20260506_203453.jsonl` | 63116 | 2026-05-06T22:32:48 |
| `strategyqa_benchmark_20260506_181806.jsonl` | 64229 | 2026-05-06T20:05:04 |
| `strategyqa_benchmark_20260506_154054.jsonl` | 63608 | 2026-05-06T17:28:19 |
| `strategyqa_benchmark_20260505_070137.jsonl` | 54385 | 2026-05-06T11:37:37 |
| `strategyqa_benchmark_20260504_103134.jsonl` | 54653 | 2026-05-05T01:26:48 |
| `strategyqa_benchmark_20260504_100626.jsonl` | 16911 | 2026-05-04T10:28:33 |
| `strategyqa_benchmark_20260504_095810.jsonl` | 3892 | 2026-05-04T10:00:08 |
| `strategyqa_benchmark_20260419_173019.jsonl` | 22491 | 2026-04-19T17:50:19 |
| `strategyqa_benchmark_20260419_172100.jsonl` | 5267 | 2026-04-19T17:29:53 |
| `strategyqa_benchmark_20260417_231238.jsonl` | 27757 | 2026-04-18T02:00:45 |
| `strategyqa_benchmark_20260417_082650.jsonl` | 6688 | 2026-04-17T08:35:49 |
| `strategyqa_benchmark_20260416_141537.jsonl` | 27757 | 2026-04-17T01:01:22 |
| `strategyqa_benchmark_20260416_105654.jsonl` | 27756 | 2026-04-16T14:01:51 |
| `strategyqa_benchmark_20260415_210431.jsonl` | 27751 | 2026-04-16T10:46:53 |
| `strategyqa_benchmark_20260415_155845.jsonl` | 4805 | 2026-04-15T17:20:34 |
| `strategyqa_benchmark_20260415_090706.jsonl` | 27749 | 2026-04-15T11:55:03 |
| `strategyqa_benchmark_20260415_000224.jsonl` | 27759 | 2026-04-15T06:48:04 |
| `strategyqa_benchmark_20260414_213252.jsonl` | 27751 | 2026-04-14T23:53:02 |
| `strategyqa_benchmark_20260414_200850.jsonl` | 27762 | 2026-04-14T20:52:22 |
| `strategyqa_benchmark_20260414_160235.jsonl` | 27746 | 2026-04-14T19:41:34 |
| `strategyqa_benchmark_20260414_120824.jsonl` | 4810 | 2026-04-14T12:53:33 |
| `strategyqa_benchmark_20260414_095001.jsonl` | 21697 | 2026-04-14T11:43:30 |
| `strategyqa_benchmark_20260414_091401.jsonl` | 1752 | 2026-04-14T09:42:39 |
| `strategyqa_benchmark_20260406_003738.jsonl` | 43342 | 2026-04-06T10:53:30 |
| `strategyqa_benchmark_20260405_220612.jsonl` | 43343 | 2026-04-06T00:17:56 |
| `strategyqa_benchmark_20260403_154639.jsonl` | 1441 | 2026-04-03T16:03:31 |
| `strategyqa_benchmark_20260401_224314.jsonl` | 43328 | 2026-04-02T01:33:59 |
| `strategyqa_benchmark_20260331_185246.jsonl` | 43346 | 2026-03-31T20:19:43 |
| `strategyqa_benchmark_20260331_184635.jsonl` | 346 | 2026-03-31T18:48:34 |
| `strategyqa_benchmark_20260330_023918_modif.jsonl` | 43321 | 2026-03-30T20:32:14 |
| `strategyqa_benchmark_20260327_210850.jsonl` | 43383 | 2026-03-27T21:23:35 |
| `strategyqa_benchmark_20260327_205345.jsonl` | 723 | 2026-03-27T21:03:01 |
| `strategyqa_benchmark_20260327_133210.jsonl` | 1441 | 2026-03-27T13:48:00 |
| `strategyqa_benchmark_20260327_115954.jsonl` | 344 | 2026-03-27T12:08:35 |
| `strategyqa_benchmark_20260327_094143.jsonl` | 345 | 2026-03-27T09:48:16 |
| `strategyqa_benchmark_20260310_184227.jsonl` | 36369 | 2026-03-10T23:38:15 |
| `strategyqa_benchmark_20260310_130459.jsonl` | 43338 | 2026-03-10T14:19:47 |
| `strategyqa_benchmark_20260310_105047.jsonl` | 26349 | 2026-03-10T13:02:47 |

### Trace JSONLs

| File | Size (bytes) | mtime (local) |
|------|-------------|---------------|
| `trace_20260506_203448_3587027.jsonl` | 1308113 | 2026-05-06T22:32:35 |
| `trace_20260506_181749_3586864.jsonl` | 1336682 | 2026-05-06T20:04:37 |
| `trace_20260506_154034_3586274.jsonl` | 1315393 | 2026-05-06T17:27:37 |
| `trace_20260505_070131_955675.jsonl` | 1858955 | 2026-05-06T11:37:37 |
| `trace_20260504_103133_3721139.jsonl` | 1800106 | 2026-05-05T01:26:48 |
| `trace_20260504_103011_3721057.jsonl` | 1906 | 2026-05-04T10:30:56 |
| `trace_20260504_100618_3720896.jsonl` | 404203 | 2026-05-04T10:28:33 |
| `trace_20260504_095748_2970181.jsonl` | 125349 | 2026-05-04T10:00:21 |
| `trace_20260423_001901_4141785.jsonl` | 151657 | 2026-04-23T00:25:28 |
| `trace_20260422_215538_4136938.jsonl` | 616378 | 2026-04-23T00:08:40 |
| `trace_20260422_215200_4095903.jsonl` | 16113 | 2026-04-22T21:52:40 |
| `trace_20260419_172019_2728173.jsonl` | 743706 | 2026-04-19T17:50:18 |
| `trace_20260417_231227_1395497.jsonl` | 1175277 | 2026-04-18T02:00:45 |
| `trace_20260417_083811_1346145.jsonl` | 263 | 2026-04-17T08:38:11 |
| `trace_20260417_082339_1346019.jsonl` | 72732 | 2026-04-17T08:36:19 |
| `trace_20260416_141534_911892.jsonl` | 2380170 | 2026-04-17T01:01:22 |
| `trace_20260416_105653_911678.jsonl` | 1211749 | 2026-04-16T14:01:51 |
| `trace_20260415_210407_452897.jsonl` | 3516394 | 2026-04-16T10:46:53 |
| `trace_20260415_090658_221385.jsonl` | 1081874 | 2026-04-15T11:55:03 |
| `trace_20260415_000218_3275869.jsonl` | 1136803 | 2026-04-15T06:48:04 |
| `trace_20260414_213239_3275637.jsonl` | 1094949 | 2026-04-14T23:53:02 |
| `trace_20260414_200844_3218330.jsonl` | 1372963 | 2026-04-14T20:52:21 |
| `trace_20260414_160229_3194264.jsonl` | 1460543 | 2026-04-14T19:41:33 |
| `trace_20260414_120819_2819312.jsonl` | 320925 | 2026-04-14T15:08:14 |
| `trace_20260414_094345_2284928.jsonl` | 818845 | 2026-04-14T11:43:30 |
| `trace_20260414_090819_2283470.jsonl` | 262405 | 2026-04-14T09:42:52 |
| `trace_20260409_174559_372695.jsonl` | 13963954 | 2026-04-14T08:01:36 |
| `trace_20260409_111254_17947.jsonl` | 32077 | 2026-04-09T11:25:27 |
| `trace_20260409_103307_17662.jsonl` | 8968 | 2026-04-09T10:33:23 |
| `trace_20260409_085917_30366.jsonl` | 355080 | 2026-04-09T10:17:16 |
| `trace_20260408_231423_24798.jsonl` | 32060 | 2026-04-08T23:30:33 |
| `trace_20260408_225241_16141.jsonl` | 41529 | 2026-04-08T23:13:43 |
| `trace_20260408_193745_6681.jsonl` | 2284597 | 2026-04-08T22:41:22 |
| `trace_20260408_184730_14268.jsonl` | 2014 | 2026-04-08T18:48:00 |
| `trace_20260407_175308_6622.jsonl` | 2501569 | 2026-04-08T18:31:20 |
| `trace_20260407_153148_15851.jsonl` | 174831 | 2026-04-07T16:31:12 |
| `trace_20260407_144559_12376.jsonl` | 43029 | 2026-04-07T14:56:38 |
| `trace_20260407_104952_2254.jsonl` | 783474 | 2026-04-07T14:31:51 |
| `trace_20260407_012530_12511.jsonl` | 1874729 | 2026-04-07T09:36:07 |
| `trace_20260406_003728_13082.jsonl` | 1815078 | 2026-04-06T10:53:28 |
| `trace_20260405_220521_32321.jsonl` | 460171 | 2026-04-06T00:03:11 |
| `trace_20260403_162818_12757.jsonl` | 5232955 | 2026-04-05T20:27:16 |
| `trace_20260403_154611_1063.jsonl` | 67399 | 2026-04-03T16:10:06 |
| `trace_20260403_152800_1145.jsonl` | 21905 | 2026-04-03T15:41:46 |
| `trace_20260403_145429_31690.jsonl` | 56550 | 2026-04-03T15:21:01 |
| `trace_20260403_133747_8971.jsonl` | 116365 | 2026-04-03T14:40:48 |
| `trace_20260403_130606_16287.jsonl` | 15282 | 2026-04-03T13:27:42 |
| `trace_20260402_120007_21877.jsonl` | 163651 | 2026-04-02T15:30:27 |
| `trace_20260402_075519_21517.jsonl` | 164134 | 2026-04-02T11:58:51 |
| `trace_20260401_223708_20974.jsonl` | 95172 | 2026-04-02T01:23:35 |
| `trace_20260401_075854_1254.jsonl` | 548195 | 2026-04-01T22:29:21 |
| `trace_20260331_203532_6212.jsonl` | 856754 | 2026-04-01T02:31:07 |
| `trace_20260331_185237_5942.jsonl` | 257817 | 2026-03-31T20:34:52 |
| `trace_20260331_184624_28579.jsonl` | 5099 | 2026-03-31T18:50:10 |
| `trace_20260331_181306_11704.jsonl` | 257 | 2026-03-31T18:13:06 |
| `trace_20260331_154643_30413.jsonl` | 93961 | 2026-03-31T16:52:31 |
| `trace_20260331_123902_2869.jsonl` | 178613 | 2026-03-31T14:34:13 |
| `trace_20260331_091517_26199.jsonl` | 125946 | 2026-03-31T12:19:13 |
| `trace_20260331_090921_21552.jsonl` | 467 | 2026-03-31T09:12:06 |
| `trace_20260328_052356_4854.jsonl` | 1747577 | 2026-03-31T08:51:31 |
| `trace_20260327_221529_7617.jsonl` | 37069 | 2026-03-27T23:27:56 |
| `trace_20260327_210845_22295.jsonl` | 1971 | 2026-03-27T21:13:07 |
| `trace_20260327_205334_17592.jsonl` | 4894 | 2026-03-27T21:03:00 |

## File pairings

| Ablation | Benchmark JSONL | Trace JSONL | Classification |
|----------|-----------------|-------------|----------------|
| chatgpt | `strategyqa_benchmark_20260506_181806.jsonl` | `trace_20260506_181749_3586864.jsonl` | `chatgpt` |
| chatgpt | `strategyqa_benchmark_20260506_203453.jsonl` | `trace_20260506_203448_3587027.jsonl` | `chatgpt` |
| chatgpt | `strategyqa_benchmark_20260506_154054.jsonl` | `trace_20260506_154034_3586274.jsonl` | `chatgpt` |
| small | `strategyqa_benchmark_20260505_070137.jsonl` | `trace_20260505_070131_955675.jsonl` | `small` |
| small | `strategyqa_benchmark_20260504_103134.jsonl` | `trace_20260504_103133_3721139.jsonl` | `small` |
| small | `strategyqa_benchmark_20260415_210431.jsonl` | `trace_20260415_210407_452897.jsonl` | `small` |

## Warnings / discovery

- no trace <= strategyqa_benchmark_20260310_105047.jsonl; skipped
- no trace <= strategyqa_benchmark_20260310_130459.jsonl; skipped
- no trace <= strategyqa_benchmark_20260310_184227.jsonl; skipped
- no trace <= strategyqa_benchmark_20260327_094143.jsonl; skipped
- no trace <= strategyqa_benchmark_20260327_115954.jsonl; skipped
- no trace <= strategyqa_benchmark_20260327_133210.jsonl; skipped
- skip benchmark (bad name): strategyqa_benchmark_20260330_023918_modif.jsonl
- strategyqa_benchmark_20260417_082650.jsonl: unknown worker models {}
- chatgpt greedy cover still missing 2 canonical StrategyQA ids among 0..299 (first gaps: [157, 229])
- small-model greedy cover missing 52 canonical ids in 0..299 (first gaps: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] …)
- Picked traces with `ceo_turn_delta_*`: ['trace_20260505_070131_955675.jsonl', 'trace_20260504_103133_3721139.jsonl']; picked runs lacking ceo token fields: ['trace_20260415_210407_452897.jsonl'].
## Question coverage

### ChatGPT ablation

- `strategyqa_benchmark_20260506_181806.jsonl`: n=100, q=[1..100], acc=0.8100
- `strategyqa_benchmark_20260506_203453.jsonl`: n=100, q=[1..100], acc=0.6300
- `strategyqa_benchmark_20260506_154054.jsonl`: n=100, q=[1..100], acc=0.6800

- Union covers StrategyQA canonical ids (trace `question_id`) min=0 max=299, count=298 (target 0..299 = first 300 rows; human ordinal +1)
- Missing canonical ids in 0..299 (first 40 shown): [157, 229]


### Small-model ablation

- `strategyqa_benchmark_20260505_070137.jsonl`: n=100, q=[1..100], acc=0.4100
- `strategyqa_benchmark_20260504_103134.jsonl`: n=100, q=[1..100], acc=0.5800
- `strategyqa_benchmark_20260415_210431.jsonl`: n=50, q=[1..50], acc=0.7200

- Union covers StrategyQA canonical ids (trace `question_id`) min=40 max=299, count=248 (target 0..299 = first 300 rows; human ordinal +1)
- Missing canonical ids in 0..299 (first 40 shown): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] …


## Accuracy (merged, StrategyQA canonical ids 0..299)

| Ablation | Correct | Total | Overall % | Excluded (wrong+429) | Adjusted denom | Adjusted % |
|----------|---------|-------|-----------|----------------------|----------------|------------|
| chatgpt | 210 | 298 | 70.47% | 24 | 274 | 76.64% |
| small | 133 | 248 | 53.63% | 72 | 176 | 75.57% |

## Per-run accuracy (full file)

### ChatGPT runs
| File | N | Correct | Acc |
|------|---|---------|-----|
| `strategyqa_benchmark_20260506_203453.jsonl` | 100 | 63 | 63.00% |
| `strategyqa_benchmark_20260506_181806.jsonl` | 100 | 81 | 81.00% |
| `strategyqa_benchmark_20260506_154054.jsonl` | 100 | 68 | 68.00% |

### Per-run — canonical ids 0..299 only (overall vs 429-filtered)
| File | Correct | Total | Overall % | Excl (wrong∧429) | Adj denom | Adj % |
|------|---------|-------|-----------|------------------|-----------|-------|
| `strategyqa_benchmark_20260506_203453.jsonl` | 62 | 99 | 62.63% | 10 | 89 | 69.66% |
| `strategyqa_benchmark_20260506_181806.jsonl` | 81 | 100 | 81.00% | 2 | 98 | 82.65% |
| `strategyqa_benchmark_20260506_154054.jsonl` | 67 | 99 | 67.68% | 12 | 87 | 77.01% |

### Small-model runs
| File | N | Correct | Acc |
|------|---|---------|-----|
| `strategyqa_benchmark_20260505_070137.jsonl` | 100 | 41 | 41.00% |
| `strategyqa_benchmark_20260504_103134.jsonl` | 100 | 58 | 58.00% |
| `strategyqa_benchmark_20260415_210431.jsonl` | 50 | 36 | 72.00% |

### Small-model runs — canonical ids 0..299 only
| File | Correct | Total | Overall % | Excl (wrong∧429) | Adj denom | Adj % |
|------|---------|-------|-----------|------------------|-----------|-------|
| `strategyqa_benchmark_20260505_070137.jsonl` | 40 | 99 | 40.40% | 45 | 54 | 74.07% |
| `strategyqa_benchmark_20260504_103134.jsonl` | 57 | 99 | 57.58% | 27 | 72 | 79.17% |
| `strategyqa_benchmark_20260415_210431.jsonl` | 36 | 50 | 72.00% | 0 | 50 | 72.00% |
## CEO token deltas per StrategyQA canonical question (merged 0..299)
### ChatGPT

- Rows with token deltas: **298**; skipped (missing CEO token data): **0** (target slice: 300 canonical ids)

**Per trace token-field presence (ceo_final_answer / strategyqa)**

- `trace_20260506_154034_3586274.jsonl`: ceo_turn_delta_*, ceo_session_*
- `trace_20260506_181749_3586864.jsonl`: ceo_turn_delta_*, ceo_session_*
- `trace_20260506_203448_3587027.jsonl`: ceo_turn_delta_*, ceo_session_*

| Metric | mean | median | p25 | p75 |
|--------|------|--------|-----|-----|
| Δ input | 109269.76 | 84381.00 | 44521.25 | 147647.25 |
| Δ output | 611.27 | 598.50 | 420.25 | 795.50 |
| Δ total | 109881.02 | 84805.00 | 45416.75 | 148279.50 |

### Small open-weight

- Rows with token deltas: **198**; skipped (missing CEO token data): **50** (target slice: 300 canonical ids)

**Per trace token-field presence (ceo_final_answer / strategyqa)**

- `trace_20260415_210407_452897.jsonl`: (no token fields on ceo_final_answer)
- `trace_20260504_103133_3721139.jsonl`: ceo_turn_delta_*, ceo_session_*
- `trace_20260505_070131_955675.jsonl`: ceo_turn_delta_*, ceo_session_*

| Metric | mean | median | p25 | p75 |
|--------|------|--------|-----|-----|
| Δ input | 104683.52 | 88570.00 | 56483.50 | 139967.75 |
| Δ output | 1839.50 | 2067.50 | 8.00 | 2562.75 |
| Δ total | 106523.02 | 91084.00 | 57449.75 | 141763.75 |

## Cumulative `ceo_session_*` vs per-question deltas

In `GeminiManager`, `self.input_tokens` and `self.output_tokens` start at 0 and only increase via `+=` (see `HASHIRU_modified/src/manager/manager.py`), so traced `ceo_session_*` snapshots are cumulative process-lifetime totals, not per-question.

### Empirical check — chatgpt
- `trace_20260506_203448_3587027.jsonl`: input_monotonic=True, output_monotonic=True
  - session_input_tokens: n=127, monotonic_nondecreasing=True, first=3680, last=22626562
  - session_output_tokens: n=127, monotonic_nondecreasing=True, first=721, last=71083
- `trace_20260506_181749_3586864.jsonl`: input_monotonic=True, output_monotonic=True
  - session_input_tokens: n=114, monotonic_nondecreasing=True, first=6041, last=11144450
  - session_output_tokens: n=114, monotonic_nondecreasing=True, first=737, last=63316
- `trace_20260506_154034_3586274.jsonl`: input_monotonic=True, output_monotonic=True
  - session_input_tokens: n=125, monotonic_nondecreasing=True, first=4876, last=12793022
  - session_output_tokens: n=125, monotonic_nondecreasing=True, first=920, last=53861

### Empirical check — small
- `trace_20260505_070131_955675.jsonl`: input_monotonic=True, output_monotonic=True
  - session_input_tokens: n=193, monotonic_nondecreasing=True, first=12977, last=18523970
  - session_output_tokens: n=193, monotonic_nondecreasing=True, first=2026, last=184825
- `trace_20260504_103133_3721139.jsonl`: input_monotonic=True, output_monotonic=True
  - session_input_tokens: n=156, monotonic_nondecreasing=True, first=12630, last=17266365
  - session_output_tokens: n=156, monotonic_nondecreasing=True, first=1980, last=199261
- `trace_20260415_210407_452897.jsonl`: input_monotonic=True, output_monotonic=True

## Caveats

- **429 / quota regex:** `(?i)(?:RESOURCE_EXHAUSTED|quota.?exhaust|exceeded.?your.?current.?quota|rate.?limits?|HTTP\s*\/?\s*429)|(?:(?<![.\d])429(?!\d))` applied to `json.dumps(event)` per trace event with `benchmark_name=strategyqa` and matching `question_index`, plus benchmark record JSON.
- **Dedup:** Merged keyed by trace `question_id` (= StrategyQA canonical id). Newer benchmark runs overwrite older ones for the same id.
- **Token skips:** Questions outside a run’s file, or `ceo_final_answer` rows without both delta and computable session diffs, are omitted from token statistics.