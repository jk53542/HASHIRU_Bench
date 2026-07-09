#!/usr/bin/env bash
set -euo pipefail

# Run deterministic StrategyQA ablations by restarting HASHIRU per condition.
#
# Why this script exists:
# - StrategyQA now supports --num-questions and --offset.
# - For true ablations you often need to restart the app with different env vars.
#
# Usage (example):
#   ./run_strategyqa_ablations.sh \
#     --start-cmd "python /path/to/hashiru_app.py" \
#     --stop-cmd "pkill -f hashiru_app.py" \
#     --conditions-file "./strategyqa_ablation_conditions.example.txt" \
#     --num-questions 20 \
#     --offset 40
#
# Conditions file format (one per line):
#   name|ENV_A=1 ENV_B=foo ENV_C=bar
# Blank lines and lines starting with # are ignored.

START_CMD=""
STOP_CMD=""
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:7860/}"
HEALTH_TIMEOUT_S=240
HEALTH_INTERVAL_S=2
BENCH_SPLIT="test"
BENCH_NUM_QUESTIONS=20
BENCH_OFFSET=0
BENCH_SCRIPT="benchmarking_strategyQA.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDITIONS_FILE=""
OUT_ROOT=""
KEEP_SERVER_UP=0
DRY_RUN=0

usage() {
  cat <<'EOF'
run_strategyqa_ablations.sh

Required:
  --start-cmd <cmd>          Command to start HASHIRU app (runs in background).
  --conditions-file <path>   File listing named ablation conditions.

Optional:
  --stop-cmd <cmd>           Command to stop HASHIRU before each run.
  --health-url <url>         Health endpoint to probe (default: http://127.0.0.1:7860/).
  --health-timeout <sec>     Max seconds waiting for health endpoint (default: 240).
  --health-interval <sec>    Poll interval in seconds (default: 2).
  --split <name>             StrategyQA split (default: test).
  --num-questions <n>        Number of questions per run (default: 20).
  --offset <n>               Start offset (default: 0).
  --bench-script <path>      StrategyQA benchmark script path (default: benchmarking_strategyQA.py).
  --python-bin <bin>         Python executable (default: python).
  --out-root <dir>           Output root dir (default: strategyqa_results/ablations_<timestamp>).
  --keep-server-up           Skip stop/start between conditions; only updates env in subprocess.
                             (Use only if your app reads env vars dynamically.)
  --dry-run                  Print commands only.
  -h, --help                 Show help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-cmd) START_CMD="$2"; shift 2 ;;
    --stop-cmd) STOP_CMD="$2"; shift 2 ;;
    --health-url) HEALTH_URL="$2"; shift 2 ;;
    --health-timeout) HEALTH_TIMEOUT_S="$2"; shift 2 ;;
    --health-interval) HEALTH_INTERVAL_S="$2"; shift 2 ;;
    --split) BENCH_SPLIT="$2"; shift 2 ;;
    --num-questions) BENCH_NUM_QUESTIONS="$2"; shift 2 ;;
    --offset) BENCH_OFFSET="$2"; shift 2 ;;
    --bench-script) BENCH_SCRIPT="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --conditions-file) CONDITIONS_FILE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --keep-server-up) KEEP_SERVER_UP=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$CONDITIONS_FILE" ]]; then
  echo "Missing required --conditions-file" >&2
  usage
  exit 1
fi
if [[ -z "$START_CMD" ]]; then
  echo "Missing required --start-cmd" >&2
  usage
  exit 1
fi
if [[ ! -f "$CONDITIONS_FILE" ]]; then
  echo "Conditions file not found: $CONDITIONS_FILE" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="strategyqa_results/ablations_${timestamp}"
fi
mkdir -p "$OUT_ROOT"
RUN_LOG="$OUT_ROOT/run.log"

echo "Ablation run started: $(date)" | tee -a "$RUN_LOG"
echo "Output root: $OUT_ROOT" | tee -a "$RUN_LOG"
echo "Conditions file: $CONDITIONS_FILE" | tee -a "$RUN_LOG"
echo "Benchmark slice: split=$BENCH_SPLIT offset=$BENCH_OFFSET num_questions=$BENCH_NUM_QUESTIONS" | tee -a "$RUN_LOG"

wait_for_health() {
  local url="$1"
  local timeout_s="$2"
  local interval_s="$3"
  local start_ts now elapsed
  start_ts="$(date +%s)"
  while true; do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    elapsed=$(( now - start_ts ))
    if (( elapsed >= timeout_s )); then
      return 1
    fi
    sleep "$interval_s"
  done
}

start_server() {
  echo "[server] starting: $START_CMD" | tee -a "$RUN_LOG"
  if (( DRY_RUN == 1 )); then
    return 0
  fi
  # Start in a subshell so env vars set by condition are applied.
  bash -lc "$START_CMD" >>"$RUN_LOG" 2>&1 &
  SERVER_PID=$!
  echo "[server] pid=$SERVER_PID" | tee -a "$RUN_LOG"
  if ! wait_for_health "$HEALTH_URL" "$HEALTH_TIMEOUT_S" "$HEALTH_INTERVAL_S"; then
    echo "[server] health check failed for $HEALTH_URL" | tee -a "$RUN_LOG"
    return 1
  fi
  echo "[server] healthy: $HEALTH_URL" | tee -a "$RUN_LOG"
}

stop_server() {
  if [[ -n "$STOP_CMD" ]]; then
    echo "[server] stopping via stop-cmd: $STOP_CMD" | tee -a "$RUN_LOG"
    if (( DRY_RUN == 0 )); then
      bash -lc "$STOP_CMD" >>"$RUN_LOG" 2>&1 || true
    fi
  elif [[ -n "${SERVER_PID:-}" ]]; then
    echo "[server] stopping pid=$SERVER_PID" | tee -a "$RUN_LOG"
    if (( DRY_RUN == 0 )); then
      kill "$SERVER_PID" >/dev/null 2>&1 || true
    fi
  fi
  sleep 2
}

run_condition() {
  local name="$1"
  local env_blob="$2"
  local safe_name
  safe_name="$(echo "$name" | tr ' ' '_' | tr -cd '[:alnum:]_-.')"
  local cond_dir="$OUT_ROOT/$safe_name"
  mkdir -p "$cond_dir"

  echo "" | tee -a "$RUN_LOG"
  echo "=== CONDITION: $name ===" | tee -a "$RUN_LOG"
  echo "env: $env_blob" | tee -a "$RUN_LOG"

  # Export condition env vars in current shell so start command sees them.
  if [[ -n "$env_blob" ]]; then
    # shellcheck disable=SC2086
    eval "export $env_blob"
  fi
  export HASHIRU_BENCH_STRATEGYQA_INTER_QUESTION_SLEEP="${HASHIRU_BENCH_STRATEGYQA_INTER_QUESTION_SLEEP:-5}"

  if (( KEEP_SERVER_UP == 0 )); then
    stop_server
    start_server
  fi

  local cmd="$PYTHON_BIN \"$BENCH_SCRIPT\" --split \"$BENCH_SPLIT\" --num-questions \"$BENCH_NUM_QUESTIONS\" --offset \"$BENCH_OFFSET\" --out-dir \"$cond_dir\""
  echo "[bench] $cmd" | tee -a "$RUN_LOG"
  if (( DRY_RUN == 1 )); then
    return 0
  fi
  if bash -lc "$cmd" >>"$RUN_LOG" 2>&1; then
    echo "[bench] condition '$name' completed" | tee -a "$RUN_LOG"
  else
    echo "[bench] condition '$name' failed (continuing to next)" | tee -a "$RUN_LOG"
  fi
}

# Optional initial start when --keep-server-up is enabled.
if (( KEEP_SERVER_UP == 1 )); then
  start_server
fi

while IFS= read -r line || [[ -n "$line" ]]; do
  # Trim leading/trailing whitespace
  line="$(echo "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  if [[ "$line" != *"|"* ]]; then
    echo "Skipping malformed condition line (missing '|'): $line" | tee -a "$RUN_LOG"
    continue
  fi
  name="${line%%|*}"
  env_blob="${line#*|}"
  run_condition "$name" "$env_blob"
done < "$CONDITIONS_FILE"

if (( KEEP_SERVER_UP == 0 )); then
  stop_server
fi

echo "Ablation run finished: $(date)" | tee -a "$RUN_LOG"
echo "Logs: $RUN_LOG"
