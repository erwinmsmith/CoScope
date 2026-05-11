#!/usr/bin/env bash
# Re-run eval_jsonl at k=5 and k=20 for every existing Stage A and Stage B
# shard set across (rpt x dataset). Reuses the workspace embedding cache,
# so query embeddings are mostly cache hits and no LLM calls are made.
#
# Outputs land next to the existing k=10 jsons:
#   data/processed/{rpt}/{dataset}/test/eval_v9_qwen_full_k{5,20}.json
#   data/processed/{rpt}/{dataset}/test_qi/eval_v9_qwen_a8_k{5,20}.json
#
# Usage: bash scripts/eval_k5_k20_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zhenke
set -a; source .env; set +a
export PYTHONHASHSEED=0

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/eval_k5_k20_sweep_${TS}.log"
mkdir -p logs

VARIANTS_FULL="a1 a2 a3 a4 a4_nofb a4_norerank a5 a5_noproj a5_norerank a6 a7"
VARIANTS_QI="a8 a6 a7"

run_eval() {
  local shard_dir="$1"      # directory holding s*.jsonl
  local out_prefix="$2"     # eval_v9_qwen_full or eval_v9_qwen_a8
  local variants="$3"
  local k="$4"

  if [[ ! -d "$shard_dir" ]]; then
    echo "  skip (missing): $shard_dir" | tee -a "$LOG"
    return
  fi
  local shards
  shards=$(ls "$shard_dir"/s*.jsonl 2>/dev/null | tr '\n' ' ')
  if [[ -z "$shards" ]]; then
    echo "  skip (no shards): $shard_dir" | tee -a "$LOG"
    return
  fi
  local out="${shard_dir}/${out_prefix}_k${k}.json"
  if [[ -f "$out" ]]; then
    echo "  exists: $out  (skip)" | tee -a "$LOG"
    return
  fi
  echo "=== eval k=$k -> $out ===" | tee -a "$LOG"
  T0=$(date +%s)
  python -m scripts.eval_jsonl \
    --shards $shards \
    --variants $variants \
    --k "$k" \
    --embedder dashscope \
    --dashscope-model text-embedding-v3 \
    --dashscope-dim 1024 \
    --cache-dir data/cache/embeddings \
    --output "$out" \
    --log-level WARNING 2>&1 | tee -a "$LOG"
  echo "    wall=$(( $(date +%s) - T0 ))s" | tee -a "$LOG"
}

# Stage A (full 11 variants) at k=5 and k=20, RPT in {got, tot}.
# (got/2wikimultihopqa is the legacy alias path; got/2wikimhqa is the canonical one.)
for k in 5 20; do
  echo "==================================================" | tee -a "$LOG"
  echo "Stage A k=$k start: $(date)"                       | tee -a "$LOG"
  echo "==================================================" | tee -a "$LOG"
  for rpt in got tot; do
    for ds in musique 2wikimhqa hotpotqa; do
      run_eval "data/processed/${rpt}/${ds}/test" \
               "eval_v9_qwen_full" "$VARIANTS_FULL" "$k"
    done
  done
  echo "Stage A k=$k done:  $(date)"                       | tee -a "$LOG"
done

# Stage B (a8/a6/a7) at k=5 and k=20, RPT in {got, tot}.
for k in 5 20; do
  echo "==================================================" | tee -a "$LOG"
  echo "Stage B k=$k start: $(date)"                       | tee -a "$LOG"
  echo "==================================================" | tee -a "$LOG"
  for rpt in got tot; do
    for ds in musique 2wikimhqa hotpotqa; do
      run_eval "data/processed/${rpt}/${ds}/test_qi" \
               "eval_v9_qwen_a8" "$VARIANTS_QI" "$k"
    done
  done
  echo "Stage B k=$k done:  $(date)"                       | tee -a "$LOG"
done

echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
