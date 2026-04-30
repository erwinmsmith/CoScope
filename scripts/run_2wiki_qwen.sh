#!/usr/bin/env bash
# Stage A-Qwen evaluation for 2WikiMultiHopQA test split (GoT, all subsets).
# k=10 first (cold, populates cache), then k=5 / k=20 (cache-warm).

set -euo pipefail
cd /home/ninghanwen/lixin/CoScope

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zhenke
set -a; source .env; set +a
export PYTHONHASHSEED=0

DATASET=2wikimultihopqa
SPLIT=test
SHARD_DIR=data/processed/got/${DATASET}/${SPLIT}
OUT_DIR=${SHARD_DIR}
mkdir -p "${OUT_DIR}"

SHARDS=(
  ${SHARD_DIR}/s1_fork.jsonl
  ${SHARD_DIR}/s2_fork.jsonl
  ${SHARD_DIR}/s2_fork_merge.jsonl
  ${SHARD_DIR}/s2_linear.jsonl
  ${SHARD_DIR}/s3_fork.jsonl
  ${SHARD_DIR}/s3_fork_merge.jsonl
  ${SHARD_DIR}/s3_independent.jsonl
  ${SHARD_DIR}/s3_linear.jsonl
  ${SHARD_DIR}/s4_policy_isolated.jsonl
)

VARIANTS="a1 a2 a3 a4 a4_nofb a4_norerank a5 a5_noproj a5_norerank a6 a7"

run_k() {
  local K=$1
  local OUT=${OUT_DIR}/eval_v9_qwen_full_k${K}.json
  echo "=== 2Wiki k=${K} start: $(date) ==="
  PYTHONHASHSEED=0 python -m scripts.eval_jsonl \
    --shards "${SHARDS[@]}" \
    --variants ${VARIANTS} \
    --k ${K} \
    --embedder dashscope \
    --dashscope-model text-embedding-v3 \
    --dashscope-dim 1024 \
    --cache-dir data/cache/embeddings \
    --output ${OUT} \
    --log-level WARNING
  echo "=== 2Wiki k=${K} done: $(date) ==="
}

run_k 10
run_k 5
run_k 20
