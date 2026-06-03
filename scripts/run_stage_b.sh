#!/usr/bin/env bash
# Stage B: generate Step-2 query_intent for all 12085 MuSiQue/GoT episodes,
# 16-way concurrent DashScope qwen-plus.
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zhenke
set -a; source .env; set +a

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/stage_b_qwen_${TS}.log
SRC=data/processed/got/musique/test
OUT=data/processed/got/musique/test_qi
mkdir -p "$OUT" logs

echo "=== Stage B query_intent start: $(date) ===" | tee -a "$LOG"
echo "src=$SRC  out=$OUT  workers=16" | tee -a "$LOG"
T0=$(date +%s)

# Pass shard files individually so the script's relative-only glob does not
# trip on absolute paths.
SHARDS=$(ls "$SRC"/*.jsonl | tr '\n' ' ')
echo "shards: $SHARDS" | tee -a "$LOG"

python -m scripts.generate_query_intent \
  --shards $SHARDS \
  --dataset musique --split test \
  --llm dashscope --model qwen-plus \
  --output-dir "$OUT" \
  --workers 32 \
  --batch-size 32 2>&1 | tee -a "$LOG"

echo "=== Stage B done: wall=$(( $(date +%s) - T0 ))s ===" | tee -a "$LOG"

echo "=== running A8 eval (k=10) on patched shards ===" | tee -a "$LOG"
T1=$(date +%s)
python -m scripts.eval_jsonl \
  --shards "$OUT"/*.jsonl \
  --variants a8 a6 a7 \
  --k 10 \
  --embedder dashscope \
  --dashscope-model text-embedding-v3 \
  --dashscope-dim 1024 \
  --cache-dir data/cache/embeddings \
  --output "$OUT/eval_v9_qwen_a8_k10.json" \
  --log-level WARNING 2>&1 | tee -a "$LOG"
echo "=== A8 eval done: wall=$(( $(date +%s) - T1 ))s ===" | tee -a "$LOG"
