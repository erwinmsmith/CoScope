#!/usr/bin/env bash
# k=20 only (cache fully warm).
set -euo pipefail
cd /home/ninghanwen/lixin/CoScope

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zhenke
set -a; source .env; set +a

export PYTHONHASHSEED=0
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/qwen_k20_${TS}.log
OUT=data/processed/got/musique/test
CACHE=data/cache/embeddings
VARIANTS="a1 a2 a3 a4 a4_nofb a4_norerank a5 a5_noproj a5_norerank a6 a7"

echo "=== k=20 Qwen eval start: $(date) ===" | tee -a "$LOG"
T0=$(date +%s)
python -m scripts.eval_jsonl \
  --shards "$OUT"/*.jsonl \
  --variants $VARIANTS \
  --k 20 \
  --embedder dashscope \
  --dashscope-model text-embedding-v3 \
  --dashscope-dim 1024 \
  --cache-dir "$CACHE" \
  --output "$OUT/eval_v9_qwen_full_k20.json" \
  --log-level WARNING 2>&1 | tee -a "$LOG"
echo "=== k=20 done: wall=$(( $(date +%s) - T0 ))s ===" | tee -a "$LOG"
