#!/usr/bin/env bash
# Build HotpotQA test shards (falls back to dev parquet since hotpot has no
# public-answer test split) using the deterministic memory_id builders, then
# run Stage A-Qwen evaluation at k=10 / k=5 / k=20.

set -euo pipefail
cd /home/ninghanwen/lixin/CoScope

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zhenke
set -a; source .env; set +a
export PYTHONHASHSEED=0

DATASET=hotpotqa
SPLIT=test
SHARD_DIR=data/processed/got/${DATASET}/${SPLIT}

echo "=== [1/2] Build ${DATASET}/${SPLIT} start: $(date) ==="
python -m scripts.build_all \
  --datasets ${DATASET} \
  --splits ${SPLIT} \
  --reasoning-path-type got \
  --max-workers 4 \
  --log-level WARNING
echo "=== [1/2] Build ${DATASET}/${SPLIT} done:  $(date) ==="

# Sanity check: abort eval if any memory_id lacks a layer prefix.
python - <<EOF
import json, glob, sys
ok = True
for path in glob.glob("${SHARD_DIR}/s*.jsonl"):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 20: break
            ep = json.loads(line)
            for m in ep["memory_entries"]:
                mid = m["memory_id"]
                prefix = mid.split("_")[1] if mid.count("_") >= 2 else ""
                if prefix not in {"ws", "ts", "pr", "rs"}:
                    print("BAD memory_id", mid, "in", path); ok = False; break
            if not ok: break
    if not ok: break
sys.exit(0 if ok else 1)
EOF

mapfile -t SHARDS < <(ls ${SHARD_DIR}/s*.jsonl)
VARIANTS="a1 a2 a3 a4 a4_nofb a4_norerank a5 a5_noproj a5_norerank a6 a7"

run_k() {
  local K=$1
  local OUT=${SHARD_DIR}/eval_v9_qwen_full_k${K}.json
  echo "=== [2/2] ${DATASET} k=${K} start: $(date) ==="
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
  echo "=== [2/2] ${DATASET} k=${K} done:  $(date) ==="
}

run_k 10
run_k 5
run_k 20
