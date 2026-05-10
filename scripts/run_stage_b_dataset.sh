#!/usr/bin/env bash
# Stage B: generate Step-2 query_intent with qwen-plus, then run
# A8 / A6 / A7 retrieval evaluation at k=10. Cache (text-embedding-v3
# SQLite) is shared across rpts and datasets.
#
# Usage:
#   DATASET=2wikimhqa             bash scripts/run_stage_b_dataset.sh   # default RPT=got
#   DATASET=hotpotqa  RPT=tot     bash scripts/run_stage_b_dataset.sh
#   DATASET=musique   RPT=tot     bash scripts/run_stage_b_dataset.sh
#
# RPT defaults to 'got' for back-compat. Source shards must already exist
# under data/processed/${RPT}/${DATASET}/test (built by run_stage_a_rpt.sh
# with --include-s4 if A8 should cover S4).

set -euo pipefail
cd /home/ninghanwen/lixin/CoScope

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zhenke
set -a; source .env; set +a
export PYTHONHASHSEED=0

DATASET="${DATASET:?DATASET env var required (e.g. musique, 2wikimhqa, hotpotqa)}"
RPT="${RPT:-got}"
SPLIT=test
SRC=data/processed/${RPT}/${DATASET}/${SPLIT}
OUT=data/processed/${RPT}/${DATASET}/${SPLIT}_qi
mkdir -p "$OUT" logs

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/stage_b_${RPT}_${DATASET}_${TS}.log

echo "=== Stage B query_intent start: $(date) ===" | tee -a "$LOG"
echo "rpt=$RPT  dataset=$DATASET  src=$SRC  out=$OUT  workers=32" | tee -a "$LOG"
if [[ ! -d "$SRC" ]]; then
  echo "ERROR: source shard dir does not exist: $SRC" | tee -a "$LOG"; exit 1
fi
T0=$(date +%s)

SHARDS=$(ls "$SRC"/s*.jsonl | tr '\n' ' ')
echo "shards: $SHARDS" | tee -a "$LOG"

python -m scripts.generate_query_intent \
  --shards $SHARDS \
  --dataset "$DATASET" --split "$SPLIT" \
  --llm dashscope --model qwen-plus \
  --output-dir "$OUT" \
  --workers 32 \
  --batch-size 32 2>&1 | tee -a "$LOG"

echo "=== Stage B query_intent done: wall=$(( $(date +%s) - T0 ))s ===" | tee -a "$LOG"

echo "=== A8 eval (k=10) start: $(date) ===" | tee -a "$LOG"
T1=$(date +%s)
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
echo "=== A8 eval done: wall=$(( $(date +%s) - T1 ))s ===" | tee -a "$LOG"
