# Benchmark run inventory (auto-generated)

## Orchestration traces (semantic mode inferred)

Inference rule on `ceo_tool_finished` rows for AskAgent/AskMultipleAgents: count rows where both `semantic_entropy` and `semantic_density` keys exist; if ≥25% have both numeric → `both_metrics_likely`; if ≥55% have both null → `neither_metrics_likely`; else entropy-only / density-only / mixed.

- `trace_20260417_231227_1395497.jsonl` → **entropy_only_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 49}
  - metric row counts: {'both_nn': 0, 'both_null': 8, 'e_only': 86, 'd_only': 0, 'other': 0}
- `trace_20260417_083811_1346145.jsonl` → **mixed_or_sparse** | benchmarks=[]
  - metric row counts: {'both_nn': 0, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260417_082339_1346019.jsonl` → **mixed_or_sparse** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 13}
  - metric row counts: {'both_nn': 0, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260416_141534_911892.jsonl` → **entropy_only_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 48}
  - metric row counts: {'both_nn': 0, 'both_null': 9, 'e_only': 151, 'd_only': 0, 'other': 0}
- `trace_20260416_105653_911678.jsonl` → **entropy_only_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 49}
  - metric row counts: {'both_nn': 0, 'both_null': 5, 'e_only': 81, 'd_only': 0, 'other': 0}
- `trace_20260415_210407_452897.jsonl` → **density_only_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 50}
  - metric row counts: {'both_nn': 0, 'both_null': 9, 'e_only': 0, 'd_only': 192, 'other': 0}
- `trace_20260415_090658_221385.jsonl` → **density_only_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 50}
  - metric row counts: {'both_nn': 0, 'both_null': 4, 'e_only': 0, 'd_only': 61, 'other': 0}
- `trace_20260415_000218_3275869.jsonl` → **density_only_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 49}
  - metric row counts: {'both_nn': 0, 'both_null': 0, 'e_only': 0, 'd_only': 81, 'other': 0}
- `trace_20260414_213239_3275637.jsonl` → **entropy_only_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 49}
  - metric row counts: {'both_nn': 0, 'both_null': 0, 'e_only': 59, 'd_only': 0, 'other': 0}
- `trace_20260414_200844_3218330.jsonl` → **neither_metrics_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 50}
  - metric row counts: {'both_nn': 0, 'both_null': 85, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260414_160229_3194264.jsonl` → **both_metrics_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 49}
  - metric row counts: {'both_nn': 90, 'both_null': 4, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260414_120819_2819312.jsonl` → **both_metrics_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 10}
  - metric row counts: {'both_nn': 26, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260414_094345_2284928.jsonl` → **both_metrics_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 48}
  - metric row counts: {'both_nn': 46, 'both_null': 3, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260414_090819_2283470.jsonl` → **both_metrics_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 5}
  - metric row counts: {'both_nn': 11, 'both_null': 1, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260409_174559_372695.jsonl` → **both_metrics_likely** | benchmarks=['mmlu_pro']
  - unique question_ids per benchmark: {'mmlu_pro': 162}
  - metric row counts: {'both_nn': 405, 'both_null': 1, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260409_111254_17947.jsonl` → **both_metrics_likely** | benchmarks=['gsm8k']
  - unique question_ids per benchmark: {'gsm8k': 3}
  - metric row counts: {'both_nn': 3, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260409_103307_17662.jsonl` → **mixed_or_sparse** | benchmarks=['mmlu_pro']
  - unique question_ids per benchmark: {'mmlu_pro': 1}
  - metric row counts: {'both_nn': 0, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260409_085917_30366.jsonl` → **both_metrics_likely** | benchmarks=['mmlu_pro']
  - unique question_ids per benchmark: {'mmlu_pro': 2}
  - metric row counts: {'both_nn': 11, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260408_231423_24798.jsonl` → **mixed_or_sparse** | benchmarks=['mmlu_pro']
  - unique question_ids per benchmark: {'mmlu_pro': 1}
  - metric row counts: {'both_nn': 0, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260408_225241_16141.jsonl` → **mixed_or_sparse** | benchmarks=['mmlu_pro']
  - unique question_ids per benchmark: {'mmlu_pro': 1}
  - metric row counts: {'both_nn': 0, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260408_193745_6681.jsonl` → **neither_metrics_likely** | benchmarks=['gsm8k']
  - unique question_ids per benchmark: {'gsm8k': 100}
  - metric row counts: {'both_nn': 0, 'both_null': 226, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260408_184730_14268.jsonl` → **mixed_or_sparse** | benchmarks=['gsm8k']
  - unique question_ids per benchmark: {'gsm8k': 1}
  - metric row counts: {'both_nn': 0, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260407_175308_6622.jsonl` → **both_metrics_likely** | benchmarks=['gsm8k']
  - unique question_ids per benchmark: {'gsm8k': 100}
  - metric row counts: {'both_nn': 240, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260407_153148_15851.jsonl` → **both_metrics_likely** | benchmarks=['gsm8k']
  - unique question_ids per benchmark: {'gsm8k': 8}
  - metric row counts: {'both_nn': 16, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260407_144559_12376.jsonl` → **both_metrics_likely** | benchmarks=['gsm8k']
  - unique question_ids per benchmark: {'gsm8k': 3}
  - metric row counts: {'both_nn': 4, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260407_104952_2254.jsonl` → **both_metrics_likely** | benchmarks=['mmlu_pro']
  - unique question_ids per benchmark: {'mmlu_pro': 4}
  - metric row counts: {'both_nn': 22, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260407_012530_12511.jsonl` → **both_metrics_likely** | benchmarks=['truthful_qa']
  - unique question_ids per benchmark: {'truthful_qa': 91}
  - metric row counts: {'both_nn': 179, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260406_003728_13082.jsonl` → **both_metrics_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 99}
  - metric row counts: {'both_nn': 129, 'both_null': 1, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260405_220521_32321.jsonl` → **both_metrics_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 24}
  - metric row counts: {'both_nn': 33, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260403_162818_12757.jsonl` → **both_metrics_likely** | benchmarks=['jailbreakbench']
  - unique question_ids per benchmark: {'jailbreakbench': 157}
  - metric row counts: {'both_nn': 249, 'both_null': 1, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260403_154611_1063.jsonl` → **both_metrics_likely** | benchmarks=['strategyqa']
  - unique question_ids per benchmark: {'strategyqa': 5}
  - metric row counts: {'both_nn': 5, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260403_152800_1145.jsonl` → **both_metrics_likely** | benchmarks=['jailbreakbench']
  - unique question_ids per benchmark: {'jailbreakbench': 2}
  - metric row counts: {'both_nn': 2, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260403_145429_31690.jsonl` → **both_metrics_likely** | benchmarks=['truthful_qa']
  - unique question_ids per benchmark: {'truthful_qa': 2}
  - metric row counts: {'both_nn': 6, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260403_133747_8971.jsonl` → **both_metrics_likely** | benchmarks=['truthful_qa']
  - unique question_ids per benchmark: {'truthful_qa': 4}
  - metric row counts: {'both_nn': 12, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260403_130606_16287.jsonl` → **both_metrics_likely** | benchmarks=['truthful_qa']
  - unique question_ids per benchmark: {'truthful_qa': 2}
  - metric row counts: {'both_nn': 2, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260402_120007_21877.jsonl` → **both_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 58, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260402_075519_21517.jsonl` → **both_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 67, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260401_223708_20974.jsonl` → **both_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 41, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260401_075854_1254.jsonl` → **both_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 163, 'both_null': 2, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260331_203532_6212.jsonl` → **neither_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 0, 'both_null': 284, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260331_185237_5942.jsonl` → **neither_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 0, 'both_null': 118, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260331_184624_28579.jsonl` → **neither_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 0, 'both_null': 2, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260331_181306_11704.jsonl` → **mixed_or_sparse** | benchmarks=[]
  - metric row counts: {'both_nn': 0, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260331_154643_30413.jsonl` → **neither_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 0, 'both_null': 55, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260331_123902_2869.jsonl` → **neither_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 0, 'both_null': 93, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260331_091517_26199.jsonl` → **both_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 61, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260331_090921_21552.jsonl` → **mixed_or_sparse** | benchmarks=[]
  - metric row counts: {'both_nn': 0, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260328_052356_4854.jsonl` → **both_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 589, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260327_221529_7617.jsonl` → **both_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 9, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260327_210845_22295.jsonl` → **both_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 1, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}
- `trace_20260327_205334_17592.jsonl` → **both_metrics_likely** | benchmarks=[]
  - metric row counts: {'both_nn': 2, 'both_null': 0, 'e_only': 0, 'd_only': 0, 'other': 0}

## StrategyQA (`strategyqa_results/*.jsonl`)

- `strategyqa_benchmark_20260417_231238.jsonl` n=50 acc=0.5200
- `strategyqa_benchmark_20260417_082650.jsonl` n=12 acc=0.0000
- `strategyqa_benchmark_20260416_141537.jsonl` n=50 acc=0.7000
- `strategyqa_benchmark_20260416_105654.jsonl` n=50 acc=0.6400
- `strategyqa_benchmark_20260415_210431.jsonl` n=50 acc=0.7200
- `strategyqa_benchmark_20260415_155845.jsonl` n=9 acc=0.6667
- `strategyqa_benchmark_20260415_090706.jsonl` n=50 acc=0.6400
- `strategyqa_benchmark_20260415_000224.jsonl` n=50 acc=0.6600
- `strategyqa_benchmark_20260414_213252.jsonl` n=50 acc=0.7200
- `strategyqa_benchmark_20260414_200850.jsonl` n=50 acc=0.7000
- `strategyqa_benchmark_20260414_160235.jsonl` n=50 acc=0.7600
- `strategyqa_benchmark_20260414_120824.jsonl` n=9 acc=0.4444
- `strategyqa_benchmark_20260414_095001.jsonl` n=50 acc=0.7800
- `strategyqa_benchmark_20260414_091401.jsonl` n=4 acc=1.0000
- `strategyqa_benchmark_20260406_003738.jsonl` n=100 acc=0.8100

## TruthfulQA (`truthful_qa_results/*.jsonl`)

- `truthful_qa_benchmark_20260407_012538.jsonl` n=90 acc=0.0000
- `truthful_qa_benchmark_20260407_011514.jsonl` n=2 acc=0.0000
- `truthful_qa_benchmark_20260403_145457.jsonl` n=1 acc=0.0000
- `truthful_qa_benchmark_20260403_134756.jsonl` n=3 acc=0.0000
- `truthful_qa_benchmark_20260403_130704.jsonl` n=1 acc=0.0000
- `truthful_qa_benchmark_20260402_075543.jsonl` n=33 acc=0.0000
- `truthful_qa_benchmark_20260331_154656.jsonl` n=50 acc=0.0000
- `truthful_qa_benchmark_20260331_153952.jsonl` n=6 acc=0.0000
- `truthful_qa_benchmark_20260331_091542.jsonl` n=30 acc=0.0000
- `truthful_qa_benchmark_20260330_203313_modif.jsonl` n=50 acc=0.0000
- `truthful_qa_benchmark_20260327_141620.jsonl` n=16 acc=0.0000
- `truthful_qa_benchmark_20260224_151933_orig.jsonl` n=50 acc=0.0000

## JailbreakBench (`results/jailbreakbench_benchmark_*.jsonl`)

- `jailbreakbench_benchmark_20260403_163036.jsonl` n=160 acc=0.6687
- `jailbreakbench_benchmark_20260403_152818.jsonl` n=2 acc=0.0000
- `jailbreakbench_benchmark_20260401_075911.jsonl` n=137 acc=0.2482
- `jailbreakbench_benchmark_20260331_203622.jsonl` n=200 acc=0.4100
- `jailbreakbench_benchmark_20260331_202458.jsonl` n=6 acc=0.6667
- `jailbreakbench_benchmark_20260331_123917.jsonl` n=102 acc=0.8529
- `jailbreakbench_benchmark_20260331_122257.jsonl` n=6 acc=1.0000
- `jailbreakbench_benchmark_20260328_052548_modif.jsonl` n=200 acc=0.6050

## GSM8K (`eval_results/gsm8k_test_*.jsonl`)

- `gsm8k_test_20260408_193756.jsonl` n=100 acc=0.9600
- `gsm8k_test_20260407_175331.jsonl` n=100 acc=0.9300

## Paper review (`results/paper_review_benchmark_*.jsonl`)

- `paper_review_benchmark_20260319_114147.jsonl` n=92 acc=0.5109 semantic_metrics_called_true=92
- `paper_review_benchmark_20260319_095746_orig.jsonl` n=100 acc=0.4200 semantic_metrics_called_true=100
- `paper_review_benchmark_20260319_095121.jsonl` n=10 acc=0.4000 semantic_metrics_called_true=10
- `paper_review_benchmark_20260319_093716.jsonl` n=10 acc=0.0000 semantic_metrics_called_true=10
- `paper_review_benchmark_20260319_092758.jsonl` n=5 acc=0.0000 semantic_metrics_called_true=5
- `paper_review_benchmark_20260319_084035.jsonl` n=10 acc=0.0000 semantic_metrics_called_true=10
- `paper_review_benchmark_20260318_144252.jsonl` n=100 acc=0.0400 semantic_metrics_called_true=100
- `paper_review_benchmark_20260318_140622.jsonl` n=10 acc=0.0000 semantic_metrics_called_true=10
- `paper_review_benchmark_20260318_135549.jsonl` n=10 acc=0.0000 semantic_metrics_called_true=0
- `paper_review_benchmark_20260318_134920_orig.jsonl` n=10 acc=0.0000 semantic_metrics_called_true=0

## MMLU-Pro per-subject summaries (`eval_results/*_summary.json`)

- `economics_summary.json` subjects=2 macro_acc≈0.2857 total_corr=4.0 total_wrong=10.0
- `health_summary.json` subjects=2 macro_acc≈0.1500 total_corr=6.0 total_wrong=34.0
- `other_summary.json` subjects=2 macro_acc≈0.3500 total_corr=14.0 total_wrong=26.0
- `history_summary.json` subjects=2 macro_acc≈0.0500 total_corr=2.0 total_wrong=38.0
- `chemistry_summary.json` subjects=2 macro_acc≈0.0000 total_corr=0 total_wrong=40.0
- `biology_summary.json` subjects=2 macro_acc≈0.1500 total_corr=6.0 total_wrong=34.0
- `psychology_summary.json` subjects=2 macro_acc≈0.1000 total_corr=4.0 total_wrong=36.0
- `law_summary.json` subjects=2 macro_acc≈0.0000 total_corr=0 total_wrong=40.0
- `business_summary.json` subjects=2 macro_acc≈0.2000 total_corr=8.0 total_wrong=32.0

## Tau2 summary jsonl (`results/tau2_*_summary*.jsonl`)

- `tau2_airline_20260402_120149_summary.jsonl` ERROR: TAU2_DATA_DIR not set or data missing
- `tau2_airline_20260401_223435_summary.jsonl` ERROR: TAU2_DATA_DIR not set or data missing
- `tau2_telecom_20260401_223432_summary.jsonl` ERROR: TAU2_DATA_DIR not set or data missing
- `tau2_retail_20260303_173707_summary.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_retail_20260303_164817_summary.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_retail_20260303_152934_summary_orig.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_retail_20260303_152543_summary_orig.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_retail_20260303_150052_summary_orig.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_airline_20260303_144855_summary_orig.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_retail_20260303_143631_summary_orig.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_telecom_20260303_142243_summary_orig.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_telecom_20260303_134559_summary_modified.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_airline_20260303_123526_summary_modified.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_airline_20260303_101334_summary_modified.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
- `tau2_retail_20260303_090849_summary_modified.jsonl` rows=1 sample_keys=['domain', 'num_tasks', 'num_trials', 'metrics', 'simulations_count']
