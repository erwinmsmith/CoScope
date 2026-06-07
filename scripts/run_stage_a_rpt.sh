#!/usr/bin/env bash
#
# Stage A driver parameterized by reasoning_path_type.
#
# Usage:
#   DATASET=musique RPT=tot bash scripts/run_stage_a_rpt.sh
#   DATASET=2wikimhqa RPT=tot bash scripts/run_stage_a_rpt.sh
#   DATASET=hotpotqa RPT=cot bash scripts/run_stage_a_rpt.sh
#
# Env knobs:
#   DATASET    musique | 2wikimhqa | hotpotqa   (required)
#   RPT        got | cot | tot                  (required)
#   K          retrieval cutoff for the eval pass (default 10)
#   WORKERS    build-side parallelism (default 8)
#   VARIANTS   space-separated list (default: a1 a2 a3 a4 a4_nofb a4_norerank)
#   INCLUDE_S4 1 to also build POLICY_ISOLATED test shard (default 0).
#              Supported for all rpts (got/cot/tot) since the rpt-S4
#              contract was lifted; see TreeBuilder/ChainBuilder.
#
# Notes:
#   * Build is CPU-only and does not call any embedding API.
#   * Eval pass uses DashScope text-embedding-v3 with the shared
#     embedding cache at data/cache/embeddings/. Workspace memory
#     IDs (mem_ws_*) are cross-rpt stable by design (see
#     memory/builders/workspace_builder.py:86), so workspace
#     embeddings cached during the GoT runs are reused unchanged.
#     Only new query strings (per-agent retrieval queries derived
#     from the rpt-specific graph topology) trigger fresh API calls.

set -euo pipefail
cd "$(dirname "$0")/.."

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zhenke
set -a; source .env; set +a
export PYTHONHASHSEED=0

: "${DATASET:?DATASET env var required (musique|2wikimhqa|hotpotqa)}"
: "${RPT:?RPT env var required (got|cot|tot)}"
K="${K:-10}"
WORKERS="${WORKERS:-8}"
VARIANTS="${VARIANTS:-a1 a2 a3 a4 a4_nofb a4_norerank}"
INCLUDE_S4="${INCLUDE_S4:-0}"
S4_FLAG=""
if [[ "${INCLUDE_S4}" == "1" ]]; then
  S4_FLAG="--include-s4"
fi

PROC_DIR="data/processed"
SHARD_DIR="${PROC_DIR}/${RPT}/${DATASET}/test"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/stage_a_${RPT}_${DATASET}_${TS}.log"

echo "=== Stage A start: $(date) ===" | tee -a "$LOG"
echo "  DATASET=${DATASET}  RPT=${RPT}  K=${K}  WORKERS=${WORKERS}" | tee -a "$LOG"
echo "  VARIANTS=${VARIANTS}" | tee -a "$LOG"

# -- 1. Build episodes for this rpt (skip if shards already exist) --
if [[ -d "${SHARD_DIR}" && -n "$(ls -A "${SHARD_DIR}" 2>/dev/null)" ]]; then
  echo "[build] reusing existing shards in ${SHARD_DIR}" | tee -a "$LOG"
else
  echo "[build] constructing ${RPT} episodes for ${DATASET}/test" | tee -a "$LOG"
  T0=$(date +%s)
  python -m scripts.build_all \
    --datasets "${DATASET}" \
    --splits test \
    --reasoning-path-type "${RPT}" \
    --processed-dir "${PROC_DIR}" \
    --max-workers "${WORKERS}" \
    ${S4_FLAG} \
    --no-enforce-coverage \
    --log-level WARNING 2>&1 | tee -a "$LOG"
  echo "[build] done in $(( $(date +%s) - T0 ))s" | tee -a "$LOG"
fi

echo "[build] shards present:" | tee -a "$LOG"
ls -la "${SHARD_DIR}" 2>&1 | tee -a "$LOG"

# -- 2. Stage A eval (k=K) --
OUT_JSON="${SHARD_DIR}/eval_v9_qwen_full_k${K}.json"
echo "[eval] -> ${OUT_JSON}" | tee -a "$LOG"
T0=$(date +%s)
python -m scripts.eval_jsonl \
  --shards "${SHARD_DIR}"/s*.jsonl \
  --variants ${VARIANTS} \
  --k "${K}" \
  --embedder dashscope \
  --dashscope-model text-embedding-v3 \
  --dashscope-dim 1024 \
  --cache-dir data/cache/embeddings \
  --output "${OUT_JSON}" \
  --log-level WARNING 2>&1 | tee -a "$LOG"
echo "=== Stage A done: wall=$(( $(date +%s) - T0 ))s ===" | tee -a "$LOG"
