#!/usr/bin/env bash
# Build 2WikiMultiHopQA dev+test episodes (GoT, all graph types incl. POLICY_ISOLATED for test).
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zhenke
set -a; source .env; set +a

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/build_2wiki_${TS}.log
mkdir -p logs

echo "=== 2Wiki build start: $(date) ===" | tee -a "$LOG"
T0=$(date +%s)

python -m scripts.build_all \
  --datasets 2wikimhqa \
  --splits test \
  --reasoning-path-type got \
  --max-workers 4 \
  --log-level WARNING 2>&1 | tee -a "$LOG"

echo "=== 2Wiki build done: wall=$(( $(date +%s) - T0 ))s ===" | tee -a "$LOG"
ls -la data/processed/got/2wikimhqa/test/ 2>/dev/null | tee -a "$LOG"
