#!/usr/bin/env bash
# Resume k=5 and k=20 (k=10 already done; cache fully warm).
set -euo pipefail
cd /home/ninghanwen/lixin/CoScope

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zhenke
set -a; source .env; set +a

export PYTHONHASHSEED=0
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/qwen_full_resume_${TS}.log
OUT=data/processed/got/musique/test
CACHE=data/cache/embeddings
VARIANTS="a1 a2 a3 a4 a4_nofb a4_norerank a5 a5_noproj a5_norerank a6 a7"

echo "==========================================" | tee -a "$LOG"
echo "Stage A-Qwen RESUME (k=5, k=20) start: $(date)" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

for K in 5 20; do
  echo | tee -a "$LOG"
  echo ">>> [eval k=$K] Qwen v3 on 12085 ep, 11 variants (cache warm)" | tee -a "$LOG"
  T0=$(date +%s)
  python -m scripts.eval_jsonl \
    --shards "$OUT"/*.jsonl \
    --variants $VARIANTS \
    --k "$K" \
    --embedder dashscope \
    --dashscope-model text-embedding-v3 \
    --dashscope-dim 1024 \
    --cache-dir "$CACHE" \
    --output "$OUT/eval_v9_qwen_full_k${K}.json" \
    --log-level WARNING 2>&1 | tee -a "$LOG"
  T1=$(date +%s)
  echo "eval k=$K wall=$((T1-T0))s" | tee -a "$LOG"
done

echo | tee -a "$LOG"
echo "Resume done: $(date)" | tee -a "$LOG"
