#!/usr/bin/env bash
# Run only the A8 eval portion (k=10) on already-prepared test_qi shards.
# Usage: DATASET=hotpotqa bash scripts/run_a8_eval_only.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zhenke
set -a; source .env; set +a
export PYTHONHASHSEED=0

DATASET="${DATASET:?DATASET env var required}"
OUT=data/processed/got/${DATASET}/test_qi
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/a8_eval_${DATASET}_${TS}.log

echo "=== A8 eval (k=10) start: $(date) ===" | tee -a "$LOG"
T0=$(date +%s)
python -m scripts.eval_jsonl \
  --shards "$OUT"/s*.jsonl \
  --variants a8 a6 a7 \
  --k 10 \
  --embedder dashscope \
  --dashscope-model text-embedding-v3 \
  --dashscope-dim 1024 \
  --cache-dir data/cache/embeddings \
  --output "$OUT/eval_v9_qwen_a8_k10.json" \
  --log-level WARNING 2>&1 | tee -a "$LOG"
echo "=== A8 eval done: wall=$(( $(date +%s) - T0 ))s ===" | tee -a "$LOG"
